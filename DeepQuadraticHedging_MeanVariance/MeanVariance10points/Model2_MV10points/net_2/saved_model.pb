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
Fnonshared_model_1/feed_forward_sub_net_11/batch_normalization_66/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*W
shared_nameHFnonshared_model_1/feed_forward_sub_net_11/batch_normalization_66/gamma
�
Znonshared_model_1/feed_forward_sub_net_11/batch_normalization_66/gamma/Read/ReadVariableOpReadVariableOpFnonshared_model_1/feed_forward_sub_net_11/batch_normalization_66/gamma*
_output_shapes
:*
dtype0
�
Enonshared_model_1/feed_forward_sub_net_11/batch_normalization_66/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*V
shared_nameGEnonshared_model_1/feed_forward_sub_net_11/batch_normalization_66/beta
�
Ynonshared_model_1/feed_forward_sub_net_11/batch_normalization_66/beta/Read/ReadVariableOpReadVariableOpEnonshared_model_1/feed_forward_sub_net_11/batch_normalization_66/beta*
_output_shapes
:*
dtype0
�
Fnonshared_model_1/feed_forward_sub_net_11/batch_normalization_67/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*W
shared_nameHFnonshared_model_1/feed_forward_sub_net_11/batch_normalization_67/gamma
�
Znonshared_model_1/feed_forward_sub_net_11/batch_normalization_67/gamma/Read/ReadVariableOpReadVariableOpFnonshared_model_1/feed_forward_sub_net_11/batch_normalization_67/gamma*
_output_shapes
:*
dtype0
�
Enonshared_model_1/feed_forward_sub_net_11/batch_normalization_67/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*V
shared_nameGEnonshared_model_1/feed_forward_sub_net_11/batch_normalization_67/beta
�
Ynonshared_model_1/feed_forward_sub_net_11/batch_normalization_67/beta/Read/ReadVariableOpReadVariableOpEnonshared_model_1/feed_forward_sub_net_11/batch_normalization_67/beta*
_output_shapes
:*
dtype0
�
Fnonshared_model_1/feed_forward_sub_net_11/batch_normalization_68/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*W
shared_nameHFnonshared_model_1/feed_forward_sub_net_11/batch_normalization_68/gamma
�
Znonshared_model_1/feed_forward_sub_net_11/batch_normalization_68/gamma/Read/ReadVariableOpReadVariableOpFnonshared_model_1/feed_forward_sub_net_11/batch_normalization_68/gamma*
_output_shapes
:*
dtype0
�
Enonshared_model_1/feed_forward_sub_net_11/batch_normalization_68/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*V
shared_nameGEnonshared_model_1/feed_forward_sub_net_11/batch_normalization_68/beta
�
Ynonshared_model_1/feed_forward_sub_net_11/batch_normalization_68/beta/Read/ReadVariableOpReadVariableOpEnonshared_model_1/feed_forward_sub_net_11/batch_normalization_68/beta*
_output_shapes
:*
dtype0
�
Fnonshared_model_1/feed_forward_sub_net_11/batch_normalization_69/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*W
shared_nameHFnonshared_model_1/feed_forward_sub_net_11/batch_normalization_69/gamma
�
Znonshared_model_1/feed_forward_sub_net_11/batch_normalization_69/gamma/Read/ReadVariableOpReadVariableOpFnonshared_model_1/feed_forward_sub_net_11/batch_normalization_69/gamma*
_output_shapes
:*
dtype0
�
Enonshared_model_1/feed_forward_sub_net_11/batch_normalization_69/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*V
shared_nameGEnonshared_model_1/feed_forward_sub_net_11/batch_normalization_69/beta
�
Ynonshared_model_1/feed_forward_sub_net_11/batch_normalization_69/beta/Read/ReadVariableOpReadVariableOpEnonshared_model_1/feed_forward_sub_net_11/batch_normalization_69/beta*
_output_shapes
:*
dtype0
�
Fnonshared_model_1/feed_forward_sub_net_11/batch_normalization_70/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*W
shared_nameHFnonshared_model_1/feed_forward_sub_net_11/batch_normalization_70/gamma
�
Znonshared_model_1/feed_forward_sub_net_11/batch_normalization_70/gamma/Read/ReadVariableOpReadVariableOpFnonshared_model_1/feed_forward_sub_net_11/batch_normalization_70/gamma*
_output_shapes
:*
dtype0
�
Enonshared_model_1/feed_forward_sub_net_11/batch_normalization_70/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*V
shared_nameGEnonshared_model_1/feed_forward_sub_net_11/batch_normalization_70/beta
�
Ynonshared_model_1/feed_forward_sub_net_11/batch_normalization_70/beta/Read/ReadVariableOpReadVariableOpEnonshared_model_1/feed_forward_sub_net_11/batch_normalization_70/beta*
_output_shapes
:*
dtype0
�
Lnonshared_model_1/feed_forward_sub_net_11/batch_normalization_66/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*]
shared_nameNLnonshared_model_1/feed_forward_sub_net_11/batch_normalization_66/moving_mean
�
`nonshared_model_1/feed_forward_sub_net_11/batch_normalization_66/moving_mean/Read/ReadVariableOpReadVariableOpLnonshared_model_1/feed_forward_sub_net_11/batch_normalization_66/moving_mean*
_output_shapes
:*
dtype0
�
Pnonshared_model_1/feed_forward_sub_net_11/batch_normalization_66/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*a
shared_nameRPnonshared_model_1/feed_forward_sub_net_11/batch_normalization_66/moving_variance
�
dnonshared_model_1/feed_forward_sub_net_11/batch_normalization_66/moving_variance/Read/ReadVariableOpReadVariableOpPnonshared_model_1/feed_forward_sub_net_11/batch_normalization_66/moving_variance*
_output_shapes
:*
dtype0
�
Lnonshared_model_1/feed_forward_sub_net_11/batch_normalization_67/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*]
shared_nameNLnonshared_model_1/feed_forward_sub_net_11/batch_normalization_67/moving_mean
�
`nonshared_model_1/feed_forward_sub_net_11/batch_normalization_67/moving_mean/Read/ReadVariableOpReadVariableOpLnonshared_model_1/feed_forward_sub_net_11/batch_normalization_67/moving_mean*
_output_shapes
:*
dtype0
�
Pnonshared_model_1/feed_forward_sub_net_11/batch_normalization_67/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*a
shared_nameRPnonshared_model_1/feed_forward_sub_net_11/batch_normalization_67/moving_variance
�
dnonshared_model_1/feed_forward_sub_net_11/batch_normalization_67/moving_variance/Read/ReadVariableOpReadVariableOpPnonshared_model_1/feed_forward_sub_net_11/batch_normalization_67/moving_variance*
_output_shapes
:*
dtype0
�
Lnonshared_model_1/feed_forward_sub_net_11/batch_normalization_68/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*]
shared_nameNLnonshared_model_1/feed_forward_sub_net_11/batch_normalization_68/moving_mean
�
`nonshared_model_1/feed_forward_sub_net_11/batch_normalization_68/moving_mean/Read/ReadVariableOpReadVariableOpLnonshared_model_1/feed_forward_sub_net_11/batch_normalization_68/moving_mean*
_output_shapes
:*
dtype0
�
Pnonshared_model_1/feed_forward_sub_net_11/batch_normalization_68/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*a
shared_nameRPnonshared_model_1/feed_forward_sub_net_11/batch_normalization_68/moving_variance
�
dnonshared_model_1/feed_forward_sub_net_11/batch_normalization_68/moving_variance/Read/ReadVariableOpReadVariableOpPnonshared_model_1/feed_forward_sub_net_11/batch_normalization_68/moving_variance*
_output_shapes
:*
dtype0
�
Lnonshared_model_1/feed_forward_sub_net_11/batch_normalization_69/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*]
shared_nameNLnonshared_model_1/feed_forward_sub_net_11/batch_normalization_69/moving_mean
�
`nonshared_model_1/feed_forward_sub_net_11/batch_normalization_69/moving_mean/Read/ReadVariableOpReadVariableOpLnonshared_model_1/feed_forward_sub_net_11/batch_normalization_69/moving_mean*
_output_shapes
:*
dtype0
�
Pnonshared_model_1/feed_forward_sub_net_11/batch_normalization_69/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*a
shared_nameRPnonshared_model_1/feed_forward_sub_net_11/batch_normalization_69/moving_variance
�
dnonshared_model_1/feed_forward_sub_net_11/batch_normalization_69/moving_variance/Read/ReadVariableOpReadVariableOpPnonshared_model_1/feed_forward_sub_net_11/batch_normalization_69/moving_variance*
_output_shapes
:*
dtype0
�
Lnonshared_model_1/feed_forward_sub_net_11/batch_normalization_70/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*]
shared_nameNLnonshared_model_1/feed_forward_sub_net_11/batch_normalization_70/moving_mean
�
`nonshared_model_1/feed_forward_sub_net_11/batch_normalization_70/moving_mean/Read/ReadVariableOpReadVariableOpLnonshared_model_1/feed_forward_sub_net_11/batch_normalization_70/moving_mean*
_output_shapes
:*
dtype0
�
Pnonshared_model_1/feed_forward_sub_net_11/batch_normalization_70/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*a
shared_nameRPnonshared_model_1/feed_forward_sub_net_11/batch_normalization_70/moving_variance
�
dnonshared_model_1/feed_forward_sub_net_11/batch_normalization_70/moving_variance/Read/ReadVariableOpReadVariableOpPnonshared_model_1/feed_forward_sub_net_11/batch_normalization_70/moving_variance*
_output_shapes
:*
dtype0
�
9nonshared_model_1/feed_forward_sub_net_11/dense_55/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*J
shared_name;9nonshared_model_1/feed_forward_sub_net_11/dense_55/kernel
�
Mnonshared_model_1/feed_forward_sub_net_11/dense_55/kernel/Read/ReadVariableOpReadVariableOp9nonshared_model_1/feed_forward_sub_net_11/dense_55/kernel*
_output_shapes

:*
dtype0
�
9nonshared_model_1/feed_forward_sub_net_11/dense_56/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*J
shared_name;9nonshared_model_1/feed_forward_sub_net_11/dense_56/kernel
�
Mnonshared_model_1/feed_forward_sub_net_11/dense_56/kernel/Read/ReadVariableOpReadVariableOp9nonshared_model_1/feed_forward_sub_net_11/dense_56/kernel*
_output_shapes

:*
dtype0
�
9nonshared_model_1/feed_forward_sub_net_11/dense_57/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*J
shared_name;9nonshared_model_1/feed_forward_sub_net_11/dense_57/kernel
�
Mnonshared_model_1/feed_forward_sub_net_11/dense_57/kernel/Read/ReadVariableOpReadVariableOp9nonshared_model_1/feed_forward_sub_net_11/dense_57/kernel*
_output_shapes

:*
dtype0
�
9nonshared_model_1/feed_forward_sub_net_11/dense_58/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*J
shared_name;9nonshared_model_1/feed_forward_sub_net_11/dense_58/kernel
�
Mnonshared_model_1/feed_forward_sub_net_11/dense_58/kernel/Read/ReadVariableOpReadVariableOp9nonshared_model_1/feed_forward_sub_net_11/dense_58/kernel*
_output_shapes

:*
dtype0
�
9nonshared_model_1/feed_forward_sub_net_11/dense_59/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*J
shared_name;9nonshared_model_1/feed_forward_sub_net_11/dense_59/kernel
�
Mnonshared_model_1/feed_forward_sub_net_11/dense_59/kernel/Read/ReadVariableOpReadVariableOp9nonshared_model_1/feed_forward_sub_net_11/dense_59/kernel*
_output_shapes

:*
dtype0
�
7nonshared_model_1/feed_forward_sub_net_11/dense_59/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*H
shared_name97nonshared_model_1/feed_forward_sub_net_11/dense_59/bias
�
Knonshared_model_1/feed_forward_sub_net_11/dense_59/bias/Read/ReadVariableOpReadVariableOp7nonshared_model_1/feed_forward_sub_net_11/dense_59/bias*
_output_shapes
:*
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
-non_trainable_variables
regularization_losses
.layer_regularization_losses
/metrics
	variables

0layers
trainable_variables
1layer_metrics
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
��
VARIABLE_VALUEFnonshared_model_1/feed_forward_sub_net_11/batch_normalization_66/gamma&variables/0/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEEnonshared_model_1/feed_forward_sub_net_11/batch_normalization_66/beta&variables/1/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEFnonshared_model_1/feed_forward_sub_net_11/batch_normalization_67/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEEnonshared_model_1/feed_forward_sub_net_11/batch_normalization_67/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEFnonshared_model_1/feed_forward_sub_net_11/batch_normalization_68/gamma&variables/4/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEEnonshared_model_1/feed_forward_sub_net_11/batch_normalization_68/beta&variables/5/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEFnonshared_model_1/feed_forward_sub_net_11/batch_normalization_69/gamma&variables/6/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEEnonshared_model_1/feed_forward_sub_net_11/batch_normalization_69/beta&variables/7/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEFnonshared_model_1/feed_forward_sub_net_11/batch_normalization_70/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEEnonshared_model_1/feed_forward_sub_net_11/batch_normalization_70/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUELnonshared_model_1/feed_forward_sub_net_11/batch_normalization_66/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEPnonshared_model_1/feed_forward_sub_net_11/batch_normalization_66/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUELnonshared_model_1/feed_forward_sub_net_11/batch_normalization_67/moving_mean'variables/12/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEPnonshared_model_1/feed_forward_sub_net_11/batch_normalization_67/moving_variance'variables/13/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUELnonshared_model_1/feed_forward_sub_net_11/batch_normalization_68/moving_mean'variables/14/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEPnonshared_model_1/feed_forward_sub_net_11/batch_normalization_68/moving_variance'variables/15/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUELnonshared_model_1/feed_forward_sub_net_11/batch_normalization_69/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEPnonshared_model_1/feed_forward_sub_net_11/batch_normalization_69/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUELnonshared_model_1/feed_forward_sub_net_11/batch_normalization_70/moving_mean'variables/18/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEPnonshared_model_1/feed_forward_sub_net_11/batch_normalization_70/moving_variance'variables/19/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9nonshared_model_1/feed_forward_sub_net_11/dense_55/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9nonshared_model_1/feed_forward_sub_net_11/dense_56/kernel'variables/21/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9nonshared_model_1/feed_forward_sub_net_11/dense_57/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9nonshared_model_1/feed_forward_sub_net_11/dense_58/kernel'variables/23/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9nonshared_model_1/feed_forward_sub_net_11/dense_59/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE7nonshared_model_1/feed_forward_sub_net_11/dense_59/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
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
 

0
1
2
3

0
1
�
`non_trainable_variables
3regularization_losses
alayer_regularization_losses
bmetrics
4	variables

clayers
5trainable_variables
dlayer_metrics
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
enon_trainable_variables
8regularization_losses
flayer_regularization_losses
gmetrics
9	variables

hlayers
:trainable_variables
ilayer_metrics
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
jnon_trainable_variables
=regularization_losses
klayer_regularization_losses
lmetrics
>	variables

mlayers
?trainable_variables
nlayer_metrics
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
onon_trainable_variables
Bregularization_losses
player_regularization_losses
qmetrics
C	variables

rlayers
Dtrainable_variables
slayer_metrics
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
tnon_trainable_variables
Gregularization_losses
ulayer_regularization_losses
vmetrics
H	variables

wlayers
Itrainable_variables
xlayer_metrics
 
 

'0

'0
�
ynon_trainable_variables
Lregularization_losses
zlayer_regularization_losses
{metrics
M	variables

|layers
Ntrainable_variables
}layer_metrics
 

(0

(0
�
~non_trainable_variables
Pregularization_losses
layer_regularization_losses
�metrics
Q	variables
�layers
Rtrainable_variables
�layer_metrics
 

)0

)0
�
�non_trainable_variables
Tregularization_losses
 �layer_regularization_losses
�metrics
U	variables
�layers
Vtrainable_variables
�layer_metrics
 

*0

*0
�
�non_trainable_variables
Xregularization_losses
 �layer_regularization_losses
�metrics
Y	variables
�layers
Ztrainable_variables
�layer_metrics
 

+0
,1

+0
,1
�
�non_trainable_variables
\regularization_losses
 �layer_regularization_losses
�metrics
]	variables
�layers
^trainable_variables
�layer_metrics
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
 
 
z
serving_default_input_1Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1Pnonshared_model_1/feed_forward_sub_net_11/batch_normalization_66/moving_varianceFnonshared_model_1/feed_forward_sub_net_11/batch_normalization_66/gammaLnonshared_model_1/feed_forward_sub_net_11/batch_normalization_66/moving_meanEnonshared_model_1/feed_forward_sub_net_11/batch_normalization_66/beta9nonshared_model_1/feed_forward_sub_net_11/dense_55/kernelPnonshared_model_1/feed_forward_sub_net_11/batch_normalization_67/moving_varianceFnonshared_model_1/feed_forward_sub_net_11/batch_normalization_67/gammaLnonshared_model_1/feed_forward_sub_net_11/batch_normalization_67/moving_meanEnonshared_model_1/feed_forward_sub_net_11/batch_normalization_67/beta9nonshared_model_1/feed_forward_sub_net_11/dense_56/kernelPnonshared_model_1/feed_forward_sub_net_11/batch_normalization_68/moving_varianceFnonshared_model_1/feed_forward_sub_net_11/batch_normalization_68/gammaLnonshared_model_1/feed_forward_sub_net_11/batch_normalization_68/moving_meanEnonshared_model_1/feed_forward_sub_net_11/batch_normalization_68/beta9nonshared_model_1/feed_forward_sub_net_11/dense_57/kernelPnonshared_model_1/feed_forward_sub_net_11/batch_normalization_69/moving_varianceFnonshared_model_1/feed_forward_sub_net_11/batch_normalization_69/gammaLnonshared_model_1/feed_forward_sub_net_11/batch_normalization_69/moving_meanEnonshared_model_1/feed_forward_sub_net_11/batch_normalization_69/beta9nonshared_model_1/feed_forward_sub_net_11/dense_58/kernelPnonshared_model_1/feed_forward_sub_net_11/batch_normalization_70/moving_varianceFnonshared_model_1/feed_forward_sub_net_11/batch_normalization_70/gammaLnonshared_model_1/feed_forward_sub_net_11/batch_normalization_70/moving_meanEnonshared_model_1/feed_forward_sub_net_11/batch_normalization_70/beta9nonshared_model_1/feed_forward_sub_net_11/dense_59/kernel7nonshared_model_1/feed_forward_sub_net_11/dense_59/bias*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_7041134
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameZnonshared_model_1/feed_forward_sub_net_11/batch_normalization_66/gamma/Read/ReadVariableOpYnonshared_model_1/feed_forward_sub_net_11/batch_normalization_66/beta/Read/ReadVariableOpZnonshared_model_1/feed_forward_sub_net_11/batch_normalization_67/gamma/Read/ReadVariableOpYnonshared_model_1/feed_forward_sub_net_11/batch_normalization_67/beta/Read/ReadVariableOpZnonshared_model_1/feed_forward_sub_net_11/batch_normalization_68/gamma/Read/ReadVariableOpYnonshared_model_1/feed_forward_sub_net_11/batch_normalization_68/beta/Read/ReadVariableOpZnonshared_model_1/feed_forward_sub_net_11/batch_normalization_69/gamma/Read/ReadVariableOpYnonshared_model_1/feed_forward_sub_net_11/batch_normalization_69/beta/Read/ReadVariableOpZnonshared_model_1/feed_forward_sub_net_11/batch_normalization_70/gamma/Read/ReadVariableOpYnonshared_model_1/feed_forward_sub_net_11/batch_normalization_70/beta/Read/ReadVariableOp`nonshared_model_1/feed_forward_sub_net_11/batch_normalization_66/moving_mean/Read/ReadVariableOpdnonshared_model_1/feed_forward_sub_net_11/batch_normalization_66/moving_variance/Read/ReadVariableOp`nonshared_model_1/feed_forward_sub_net_11/batch_normalization_67/moving_mean/Read/ReadVariableOpdnonshared_model_1/feed_forward_sub_net_11/batch_normalization_67/moving_variance/Read/ReadVariableOp`nonshared_model_1/feed_forward_sub_net_11/batch_normalization_68/moving_mean/Read/ReadVariableOpdnonshared_model_1/feed_forward_sub_net_11/batch_normalization_68/moving_variance/Read/ReadVariableOp`nonshared_model_1/feed_forward_sub_net_11/batch_normalization_69/moving_mean/Read/ReadVariableOpdnonshared_model_1/feed_forward_sub_net_11/batch_normalization_69/moving_variance/Read/ReadVariableOp`nonshared_model_1/feed_forward_sub_net_11/batch_normalization_70/moving_mean/Read/ReadVariableOpdnonshared_model_1/feed_forward_sub_net_11/batch_normalization_70/moving_variance/Read/ReadVariableOpMnonshared_model_1/feed_forward_sub_net_11/dense_55/kernel/Read/ReadVariableOpMnonshared_model_1/feed_forward_sub_net_11/dense_56/kernel/Read/ReadVariableOpMnonshared_model_1/feed_forward_sub_net_11/dense_57/kernel/Read/ReadVariableOpMnonshared_model_1/feed_forward_sub_net_11/dense_58/kernel/Read/ReadVariableOpMnonshared_model_1/feed_forward_sub_net_11/dense_59/kernel/Read/ReadVariableOpKnonshared_model_1/feed_forward_sub_net_11/dense_59/bias/Read/ReadVariableOpConst*'
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
 __inference__traced_save_7042532
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameFnonshared_model_1/feed_forward_sub_net_11/batch_normalization_66/gammaEnonshared_model_1/feed_forward_sub_net_11/batch_normalization_66/betaFnonshared_model_1/feed_forward_sub_net_11/batch_normalization_67/gammaEnonshared_model_1/feed_forward_sub_net_11/batch_normalization_67/betaFnonshared_model_1/feed_forward_sub_net_11/batch_normalization_68/gammaEnonshared_model_1/feed_forward_sub_net_11/batch_normalization_68/betaFnonshared_model_1/feed_forward_sub_net_11/batch_normalization_69/gammaEnonshared_model_1/feed_forward_sub_net_11/batch_normalization_69/betaFnonshared_model_1/feed_forward_sub_net_11/batch_normalization_70/gammaEnonshared_model_1/feed_forward_sub_net_11/batch_normalization_70/betaLnonshared_model_1/feed_forward_sub_net_11/batch_normalization_66/moving_meanPnonshared_model_1/feed_forward_sub_net_11/batch_normalization_66/moving_varianceLnonshared_model_1/feed_forward_sub_net_11/batch_normalization_67/moving_meanPnonshared_model_1/feed_forward_sub_net_11/batch_normalization_67/moving_varianceLnonshared_model_1/feed_forward_sub_net_11/batch_normalization_68/moving_meanPnonshared_model_1/feed_forward_sub_net_11/batch_normalization_68/moving_varianceLnonshared_model_1/feed_forward_sub_net_11/batch_normalization_69/moving_meanPnonshared_model_1/feed_forward_sub_net_11/batch_normalization_69/moving_varianceLnonshared_model_1/feed_forward_sub_net_11/batch_normalization_70/moving_meanPnonshared_model_1/feed_forward_sub_net_11/batch_normalization_70/moving_variance9nonshared_model_1/feed_forward_sub_net_11/dense_55/kernel9nonshared_model_1/feed_forward_sub_net_11/dense_56/kernel9nonshared_model_1/feed_forward_sub_net_11/dense_57/kernel9nonshared_model_1/feed_forward_sub_net_11/dense_58/kernel9nonshared_model_1/feed_forward_sub_net_11/dense_59/kernel7nonshared_model_1/feed_forward_sub_net_11/dense_59/bias*&
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
#__inference__traced_restore_7042620��
�
�
E__inference_dense_58_layer_call_and_return_conditional_losses_7042412

inputs0
matmul_readvariableop_resource:
identity��MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:���������2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_68_layer_call_and_return_conditional_losses_7040006

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
E__inference_dense_59_layer_call_and_return_conditional_losses_7040590

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�,
�
S__inference_batch_normalization_70_layer_call_and_return_conditional_losses_7040400

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
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

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������2
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

:*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze�
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
:*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2
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
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_66_layer_call_fn_7041972

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_66_layer_call_and_return_conditional_losses_70397362
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�,
�
S__inference_batch_normalization_67_layer_call_and_return_conditional_losses_7039902

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
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

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������2
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

:*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze�
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
:*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2
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
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
~
*__inference_dense_55_layer_call_fn_7042363

inputs
unknown:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_55_layer_call_and_return_conditional_losses_70405032
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_dense_56_layer_call_and_return_conditional_losses_7042384

inputs0
matmul_readvariableop_resource:
identity��MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:���������2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�,
�
S__inference_batch_normalization_67_layer_call_and_return_conditional_losses_7042110

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
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

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������2
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

:*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze�
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
:*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2
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
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_68_layer_call_and_return_conditional_losses_7042156

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
E__inference_dense_59_layer_call_and_return_conditional_losses_7042431

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�E
�
T__inference_feed_forward_sub_net_11_layer_call_and_return_conditional_losses_7040597
x,
batch_normalization_66_7040487:,
batch_normalization_66_7040489:,
batch_normalization_66_7040491:,
batch_normalization_66_7040493:"
dense_55_7040504:,
batch_normalization_67_7040507:,
batch_normalization_67_7040509:,
batch_normalization_67_7040511:,
batch_normalization_67_7040513:"
dense_56_7040525:,
batch_normalization_68_7040528:,
batch_normalization_68_7040530:,
batch_normalization_68_7040532:,
batch_normalization_68_7040534:"
dense_57_7040546:,
batch_normalization_69_7040549:,
batch_normalization_69_7040551:,
batch_normalization_69_7040553:,
batch_normalization_69_7040555:"
dense_58_7040567:,
batch_normalization_70_7040570:,
batch_normalization_70_7040572:,
batch_normalization_70_7040574:,
batch_normalization_70_7040576:"
dense_59_7040591:
dense_59_7040593:
identity��.batch_normalization_66/StatefulPartitionedCall�.batch_normalization_67/StatefulPartitionedCall�.batch_normalization_68/StatefulPartitionedCall�.batch_normalization_69/StatefulPartitionedCall�.batch_normalization_70/StatefulPartitionedCall� dense_55/StatefulPartitionedCall� dense_56/StatefulPartitionedCall� dense_57/StatefulPartitionedCall� dense_58/StatefulPartitionedCall� dense_59/StatefulPartitionedCall�
.batch_normalization_66/StatefulPartitionedCallStatefulPartitionedCallxbatch_normalization_66_7040487batch_normalization_66_7040489batch_normalization_66_7040491batch_normalization_66_7040493*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_66_layer_call_and_return_conditional_losses_703967420
.batch_normalization_66/StatefulPartitionedCall�
 dense_55/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_66/StatefulPartitionedCall:output:0dense_55_7040504*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_55_layer_call_and_return_conditional_losses_70405032"
 dense_55/StatefulPartitionedCall�
.batch_normalization_67/StatefulPartitionedCallStatefulPartitionedCall)dense_55/StatefulPartitionedCall:output:0batch_normalization_67_7040507batch_normalization_67_7040509batch_normalization_67_7040511batch_normalization_67_7040513*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_67_layer_call_and_return_conditional_losses_703984020
.batch_normalization_67/StatefulPartitionedCall
ReluRelu7batch_normalization_67/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
Relu�
 dense_56/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0dense_56_7040525*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_56_layer_call_and_return_conditional_losses_70405242"
 dense_56/StatefulPartitionedCall�
.batch_normalization_68/StatefulPartitionedCallStatefulPartitionedCall)dense_56/StatefulPartitionedCall:output:0batch_normalization_68_7040528batch_normalization_68_7040530batch_normalization_68_7040532batch_normalization_68_7040534*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_68_layer_call_and_return_conditional_losses_704000620
.batch_normalization_68/StatefulPartitionedCall�
Relu_1Relu7batch_normalization_68/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
Relu_1�
 dense_57/StatefulPartitionedCallStatefulPartitionedCallRelu_1:activations:0dense_57_7040546*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_57_layer_call_and_return_conditional_losses_70405452"
 dense_57/StatefulPartitionedCall�
.batch_normalization_69/StatefulPartitionedCallStatefulPartitionedCall)dense_57/StatefulPartitionedCall:output:0batch_normalization_69_7040549batch_normalization_69_7040551batch_normalization_69_7040553batch_normalization_69_7040555*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_69_layer_call_and_return_conditional_losses_704017220
.batch_normalization_69/StatefulPartitionedCall�
Relu_2Relu7batch_normalization_69/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
Relu_2�
 dense_58/StatefulPartitionedCallStatefulPartitionedCallRelu_2:activations:0dense_58_7040567*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_58_layer_call_and_return_conditional_losses_70405662"
 dense_58/StatefulPartitionedCall�
.batch_normalization_70/StatefulPartitionedCallStatefulPartitionedCall)dense_58/StatefulPartitionedCall:output:0batch_normalization_70_7040570batch_normalization_70_7040572batch_normalization_70_7040574batch_normalization_70_7040576*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_70_layer_call_and_return_conditional_losses_704033820
.batch_normalization_70/StatefulPartitionedCall�
Relu_3Relu7batch_normalization_70/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
Relu_3�
 dense_59/StatefulPartitionedCallStatefulPartitionedCallRelu_3:activations:0dense_59_7040591dense_59_7040593*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_59_layer_call_and_return_conditional_losses_70405902"
 dense_59/StatefulPartitionedCall�
IdentityIdentity)dense_59/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp/^batch_normalization_66/StatefulPartitionedCall/^batch_normalization_67/StatefulPartitionedCall/^batch_normalization_68/StatefulPartitionedCall/^batch_normalization_69/StatefulPartitionedCall/^batch_normalization_70/StatefulPartitionedCall!^dense_55/StatefulPartitionedCall!^dense_56/StatefulPartitionedCall!^dense_57/StatefulPartitionedCall!^dense_58/StatefulPartitionedCall!^dense_59/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_66/StatefulPartitionedCall.batch_normalization_66/StatefulPartitionedCall2`
.batch_normalization_67/StatefulPartitionedCall.batch_normalization_67/StatefulPartitionedCall2`
.batch_normalization_68/StatefulPartitionedCall.batch_normalization_68/StatefulPartitionedCall2`
.batch_normalization_69/StatefulPartitionedCall.batch_normalization_69/StatefulPartitionedCall2`
.batch_normalization_70/StatefulPartitionedCall.batch_normalization_70/StatefulPartitionedCall2D
 dense_55/StatefulPartitionedCall dense_55/StatefulPartitionedCall2D
 dense_56/StatefulPartitionedCall dense_56/StatefulPartitionedCall2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall2D
 dense_58/StatefulPartitionedCall dense_58/StatefulPartitionedCall2D
 dense_59/StatefulPartitionedCall dense_59/StatefulPartitionedCall:J F
'
_output_shapes
:���������

_user_specified_namex
�
�
8__inference_batch_normalization_67_layer_call_fn_7042054

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_67_layer_call_and_return_conditional_losses_70399022
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
T__inference_feed_forward_sub_net_11_layer_call_and_return_conditional_losses_7041760
input_1F
8batch_normalization_66_batchnorm_readvariableop_resource:J
<batch_normalization_66_batchnorm_mul_readvariableop_resource:H
:batch_normalization_66_batchnorm_readvariableop_1_resource:H
:batch_normalization_66_batchnorm_readvariableop_2_resource:9
'dense_55_matmul_readvariableop_resource:F
8batch_normalization_67_batchnorm_readvariableop_resource:J
<batch_normalization_67_batchnorm_mul_readvariableop_resource:H
:batch_normalization_67_batchnorm_readvariableop_1_resource:H
:batch_normalization_67_batchnorm_readvariableop_2_resource:9
'dense_56_matmul_readvariableop_resource:F
8batch_normalization_68_batchnorm_readvariableop_resource:J
<batch_normalization_68_batchnorm_mul_readvariableop_resource:H
:batch_normalization_68_batchnorm_readvariableop_1_resource:H
:batch_normalization_68_batchnorm_readvariableop_2_resource:9
'dense_57_matmul_readvariableop_resource:F
8batch_normalization_69_batchnorm_readvariableop_resource:J
<batch_normalization_69_batchnorm_mul_readvariableop_resource:H
:batch_normalization_69_batchnorm_readvariableop_1_resource:H
:batch_normalization_69_batchnorm_readvariableop_2_resource:9
'dense_58_matmul_readvariableop_resource:F
8batch_normalization_70_batchnorm_readvariableop_resource:J
<batch_normalization_70_batchnorm_mul_readvariableop_resource:H
:batch_normalization_70_batchnorm_readvariableop_1_resource:H
:batch_normalization_70_batchnorm_readvariableop_2_resource:9
'dense_59_matmul_readvariableop_resource:6
(dense_59_biasadd_readvariableop_resource:
identity��/batch_normalization_66/batchnorm/ReadVariableOp�1batch_normalization_66/batchnorm/ReadVariableOp_1�1batch_normalization_66/batchnorm/ReadVariableOp_2�3batch_normalization_66/batchnorm/mul/ReadVariableOp�/batch_normalization_67/batchnorm/ReadVariableOp�1batch_normalization_67/batchnorm/ReadVariableOp_1�1batch_normalization_67/batchnorm/ReadVariableOp_2�3batch_normalization_67/batchnorm/mul/ReadVariableOp�/batch_normalization_68/batchnorm/ReadVariableOp�1batch_normalization_68/batchnorm/ReadVariableOp_1�1batch_normalization_68/batchnorm/ReadVariableOp_2�3batch_normalization_68/batchnorm/mul/ReadVariableOp�/batch_normalization_69/batchnorm/ReadVariableOp�1batch_normalization_69/batchnorm/ReadVariableOp_1�1batch_normalization_69/batchnorm/ReadVariableOp_2�3batch_normalization_69/batchnorm/mul/ReadVariableOp�/batch_normalization_70/batchnorm/ReadVariableOp�1batch_normalization_70/batchnorm/ReadVariableOp_1�1batch_normalization_70/batchnorm/ReadVariableOp_2�3batch_normalization_70/batchnorm/mul/ReadVariableOp�dense_55/MatMul/ReadVariableOp�dense_56/MatMul/ReadVariableOp�dense_57/MatMul/ReadVariableOp�dense_58/MatMul/ReadVariableOp�dense_59/BiasAdd/ReadVariableOp�dense_59/MatMul/ReadVariableOp�
/batch_normalization_66/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_66_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_66/batchnorm/ReadVariableOp�
&batch_normalization_66/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_66/batchnorm/add/y�
$batch_normalization_66/batchnorm/addAddV27batch_normalization_66/batchnorm/ReadVariableOp:value:0/batch_normalization_66/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_66/batchnorm/add�
&batch_normalization_66/batchnorm/RsqrtRsqrt(batch_normalization_66/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_66/batchnorm/Rsqrt�
3batch_normalization_66/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_66_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_66/batchnorm/mul/ReadVariableOp�
$batch_normalization_66/batchnorm/mulMul*batch_normalization_66/batchnorm/Rsqrt:y:0;batch_normalization_66/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_66/batchnorm/mul�
&batch_normalization_66/batchnorm/mul_1Mulinput_1(batch_normalization_66/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_66/batchnorm/mul_1�
1batch_normalization_66/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_66_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_66/batchnorm/ReadVariableOp_1�
&batch_normalization_66/batchnorm/mul_2Mul9batch_normalization_66/batchnorm/ReadVariableOp_1:value:0(batch_normalization_66/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_66/batchnorm/mul_2�
1batch_normalization_66/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_66_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_66/batchnorm/ReadVariableOp_2�
$batch_normalization_66/batchnorm/subSub9batch_normalization_66/batchnorm/ReadVariableOp_2:value:0*batch_normalization_66/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_66/batchnorm/sub�
&batch_normalization_66/batchnorm/add_1AddV2*batch_normalization_66/batchnorm/mul_1:z:0(batch_normalization_66/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_66/batchnorm/add_1�
dense_55/MatMul/ReadVariableOpReadVariableOp'dense_55_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_55/MatMul/ReadVariableOp�
dense_55/MatMulMatMul*batch_normalization_66/batchnorm/add_1:z:0&dense_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_55/MatMul�
/batch_normalization_67/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_67_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_67/batchnorm/ReadVariableOp�
&batch_normalization_67/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_67/batchnorm/add/y�
$batch_normalization_67/batchnorm/addAddV27batch_normalization_67/batchnorm/ReadVariableOp:value:0/batch_normalization_67/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_67/batchnorm/add�
&batch_normalization_67/batchnorm/RsqrtRsqrt(batch_normalization_67/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_67/batchnorm/Rsqrt�
3batch_normalization_67/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_67_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_67/batchnorm/mul/ReadVariableOp�
$batch_normalization_67/batchnorm/mulMul*batch_normalization_67/batchnorm/Rsqrt:y:0;batch_normalization_67/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_67/batchnorm/mul�
&batch_normalization_67/batchnorm/mul_1Muldense_55/MatMul:product:0(batch_normalization_67/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_67/batchnorm/mul_1�
1batch_normalization_67/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_67_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_67/batchnorm/ReadVariableOp_1�
&batch_normalization_67/batchnorm/mul_2Mul9batch_normalization_67/batchnorm/ReadVariableOp_1:value:0(batch_normalization_67/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_67/batchnorm/mul_2�
1batch_normalization_67/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_67_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_67/batchnorm/ReadVariableOp_2�
$batch_normalization_67/batchnorm/subSub9batch_normalization_67/batchnorm/ReadVariableOp_2:value:0*batch_normalization_67/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_67/batchnorm/sub�
&batch_normalization_67/batchnorm/add_1AddV2*batch_normalization_67/batchnorm/mul_1:z:0(batch_normalization_67/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_67/batchnorm/add_1r
ReluRelu*batch_normalization_67/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu�
dense_56/MatMul/ReadVariableOpReadVariableOp'dense_56_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_56/MatMul/ReadVariableOp�
dense_56/MatMulMatMulRelu:activations:0&dense_56/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_56/MatMul�
/batch_normalization_68/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_68_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_68/batchnorm/ReadVariableOp�
&batch_normalization_68/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_68/batchnorm/add/y�
$batch_normalization_68/batchnorm/addAddV27batch_normalization_68/batchnorm/ReadVariableOp:value:0/batch_normalization_68/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_68/batchnorm/add�
&batch_normalization_68/batchnorm/RsqrtRsqrt(batch_normalization_68/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_68/batchnorm/Rsqrt�
3batch_normalization_68/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_68_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_68/batchnorm/mul/ReadVariableOp�
$batch_normalization_68/batchnorm/mulMul*batch_normalization_68/batchnorm/Rsqrt:y:0;batch_normalization_68/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_68/batchnorm/mul�
&batch_normalization_68/batchnorm/mul_1Muldense_56/MatMul:product:0(batch_normalization_68/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_68/batchnorm/mul_1�
1batch_normalization_68/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_68_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_68/batchnorm/ReadVariableOp_1�
&batch_normalization_68/batchnorm/mul_2Mul9batch_normalization_68/batchnorm/ReadVariableOp_1:value:0(batch_normalization_68/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_68/batchnorm/mul_2�
1batch_normalization_68/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_68_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_68/batchnorm/ReadVariableOp_2�
$batch_normalization_68/batchnorm/subSub9batch_normalization_68/batchnorm/ReadVariableOp_2:value:0*batch_normalization_68/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_68/batchnorm/sub�
&batch_normalization_68/batchnorm/add_1AddV2*batch_normalization_68/batchnorm/mul_1:z:0(batch_normalization_68/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_68/batchnorm/add_1v
Relu_1Relu*batch_normalization_68/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_1�
dense_57/MatMul/ReadVariableOpReadVariableOp'dense_57_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_57/MatMul/ReadVariableOp�
dense_57/MatMulMatMulRelu_1:activations:0&dense_57/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_57/MatMul�
/batch_normalization_69/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_69_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_69/batchnorm/ReadVariableOp�
&batch_normalization_69/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_69/batchnorm/add/y�
$batch_normalization_69/batchnorm/addAddV27batch_normalization_69/batchnorm/ReadVariableOp:value:0/batch_normalization_69/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_69/batchnorm/add�
&batch_normalization_69/batchnorm/RsqrtRsqrt(batch_normalization_69/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_69/batchnorm/Rsqrt�
3batch_normalization_69/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_69_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_69/batchnorm/mul/ReadVariableOp�
$batch_normalization_69/batchnorm/mulMul*batch_normalization_69/batchnorm/Rsqrt:y:0;batch_normalization_69/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_69/batchnorm/mul�
&batch_normalization_69/batchnorm/mul_1Muldense_57/MatMul:product:0(batch_normalization_69/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_69/batchnorm/mul_1�
1batch_normalization_69/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_69_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_69/batchnorm/ReadVariableOp_1�
&batch_normalization_69/batchnorm/mul_2Mul9batch_normalization_69/batchnorm/ReadVariableOp_1:value:0(batch_normalization_69/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_69/batchnorm/mul_2�
1batch_normalization_69/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_69_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_69/batchnorm/ReadVariableOp_2�
$batch_normalization_69/batchnorm/subSub9batch_normalization_69/batchnorm/ReadVariableOp_2:value:0*batch_normalization_69/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_69/batchnorm/sub�
&batch_normalization_69/batchnorm/add_1AddV2*batch_normalization_69/batchnorm/mul_1:z:0(batch_normalization_69/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_69/batchnorm/add_1v
Relu_2Relu*batch_normalization_69/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_2�
dense_58/MatMul/ReadVariableOpReadVariableOp'dense_58_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_58/MatMul/ReadVariableOp�
dense_58/MatMulMatMulRelu_2:activations:0&dense_58/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_58/MatMul�
/batch_normalization_70/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_70_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_70/batchnorm/ReadVariableOp�
&batch_normalization_70/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_70/batchnorm/add/y�
$batch_normalization_70/batchnorm/addAddV27batch_normalization_70/batchnorm/ReadVariableOp:value:0/batch_normalization_70/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_70/batchnorm/add�
&batch_normalization_70/batchnorm/RsqrtRsqrt(batch_normalization_70/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_70/batchnorm/Rsqrt�
3batch_normalization_70/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_70_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_70/batchnorm/mul/ReadVariableOp�
$batch_normalization_70/batchnorm/mulMul*batch_normalization_70/batchnorm/Rsqrt:y:0;batch_normalization_70/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_70/batchnorm/mul�
&batch_normalization_70/batchnorm/mul_1Muldense_58/MatMul:product:0(batch_normalization_70/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_70/batchnorm/mul_1�
1batch_normalization_70/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_70_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_70/batchnorm/ReadVariableOp_1�
&batch_normalization_70/batchnorm/mul_2Mul9batch_normalization_70/batchnorm/ReadVariableOp_1:value:0(batch_normalization_70/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_70/batchnorm/mul_2�
1batch_normalization_70/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_70_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_70/batchnorm/ReadVariableOp_2�
$batch_normalization_70/batchnorm/subSub9batch_normalization_70/batchnorm/ReadVariableOp_2:value:0*batch_normalization_70/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_70/batchnorm/sub�
&batch_normalization_70/batchnorm/add_1AddV2*batch_normalization_70/batchnorm/mul_1:z:0(batch_normalization_70/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_70/batchnorm/add_1v
Relu_3Relu*batch_normalization_70/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_3�
dense_59/MatMul/ReadVariableOpReadVariableOp'dense_59_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_59/MatMul/ReadVariableOp�
dense_59/MatMulMatMulRelu_3:activations:0&dense_59/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_59/MatMul�
dense_59/BiasAdd/ReadVariableOpReadVariableOp(dense_59_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_59/BiasAdd/ReadVariableOp�
dense_59/BiasAddBiasAdddense_59/MatMul:product:0'dense_59/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_59/BiasAddt
IdentityIdentitydense_59/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�

NoOpNoOp0^batch_normalization_66/batchnorm/ReadVariableOp2^batch_normalization_66/batchnorm/ReadVariableOp_12^batch_normalization_66/batchnorm/ReadVariableOp_24^batch_normalization_66/batchnorm/mul/ReadVariableOp0^batch_normalization_67/batchnorm/ReadVariableOp2^batch_normalization_67/batchnorm/ReadVariableOp_12^batch_normalization_67/batchnorm/ReadVariableOp_24^batch_normalization_67/batchnorm/mul/ReadVariableOp0^batch_normalization_68/batchnorm/ReadVariableOp2^batch_normalization_68/batchnorm/ReadVariableOp_12^batch_normalization_68/batchnorm/ReadVariableOp_24^batch_normalization_68/batchnorm/mul/ReadVariableOp0^batch_normalization_69/batchnorm/ReadVariableOp2^batch_normalization_69/batchnorm/ReadVariableOp_12^batch_normalization_69/batchnorm/ReadVariableOp_24^batch_normalization_69/batchnorm/mul/ReadVariableOp0^batch_normalization_70/batchnorm/ReadVariableOp2^batch_normalization_70/batchnorm/ReadVariableOp_12^batch_normalization_70/batchnorm/ReadVariableOp_24^batch_normalization_70/batchnorm/mul/ReadVariableOp^dense_55/MatMul/ReadVariableOp^dense_56/MatMul/ReadVariableOp^dense_57/MatMul/ReadVariableOp^dense_58/MatMul/ReadVariableOp ^dense_59/BiasAdd/ReadVariableOp^dense_59/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������: : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_66/batchnorm/ReadVariableOp/batch_normalization_66/batchnorm/ReadVariableOp2f
1batch_normalization_66/batchnorm/ReadVariableOp_11batch_normalization_66/batchnorm/ReadVariableOp_12f
1batch_normalization_66/batchnorm/ReadVariableOp_21batch_normalization_66/batchnorm/ReadVariableOp_22j
3batch_normalization_66/batchnorm/mul/ReadVariableOp3batch_normalization_66/batchnorm/mul/ReadVariableOp2b
/batch_normalization_67/batchnorm/ReadVariableOp/batch_normalization_67/batchnorm/ReadVariableOp2f
1batch_normalization_67/batchnorm/ReadVariableOp_11batch_normalization_67/batchnorm/ReadVariableOp_12f
1batch_normalization_67/batchnorm/ReadVariableOp_21batch_normalization_67/batchnorm/ReadVariableOp_22j
3batch_normalization_67/batchnorm/mul/ReadVariableOp3batch_normalization_67/batchnorm/mul/ReadVariableOp2b
/batch_normalization_68/batchnorm/ReadVariableOp/batch_normalization_68/batchnorm/ReadVariableOp2f
1batch_normalization_68/batchnorm/ReadVariableOp_11batch_normalization_68/batchnorm/ReadVariableOp_12f
1batch_normalization_68/batchnorm/ReadVariableOp_21batch_normalization_68/batchnorm/ReadVariableOp_22j
3batch_normalization_68/batchnorm/mul/ReadVariableOp3batch_normalization_68/batchnorm/mul/ReadVariableOp2b
/batch_normalization_69/batchnorm/ReadVariableOp/batch_normalization_69/batchnorm/ReadVariableOp2f
1batch_normalization_69/batchnorm/ReadVariableOp_11batch_normalization_69/batchnorm/ReadVariableOp_12f
1batch_normalization_69/batchnorm/ReadVariableOp_21batch_normalization_69/batchnorm/ReadVariableOp_22j
3batch_normalization_69/batchnorm/mul/ReadVariableOp3batch_normalization_69/batchnorm/mul/ReadVariableOp2b
/batch_normalization_70/batchnorm/ReadVariableOp/batch_normalization_70/batchnorm/ReadVariableOp2f
1batch_normalization_70/batchnorm/ReadVariableOp_11batch_normalization_70/batchnorm/ReadVariableOp_12f
1batch_normalization_70/batchnorm/ReadVariableOp_21batch_normalization_70/batchnorm/ReadVariableOp_22j
3batch_normalization_70/batchnorm/mul/ReadVariableOp3batch_normalization_70/batchnorm/mul/ReadVariableOp2@
dense_55/MatMul/ReadVariableOpdense_55/MatMul/ReadVariableOp2@
dense_56/MatMul/ReadVariableOpdense_56/MatMul/ReadVariableOp2@
dense_57/MatMul/ReadVariableOpdense_57/MatMul/ReadVariableOp2@
dense_58/MatMul/ReadVariableOpdense_58/MatMul/ReadVariableOp2B
dense_59/BiasAdd/ReadVariableOpdense_59/BiasAdd/ReadVariableOp2@
dense_59/MatMul/ReadVariableOpdense_59/MatMul/ReadVariableOp:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
8__inference_batch_normalization_66_layer_call_fn_7041959

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_66_layer_call_and_return_conditional_losses_70396742
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_68_layer_call_fn_7042123

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_68_layer_call_and_return_conditional_losses_70400062
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_67_layer_call_and_return_conditional_losses_7042074

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_dense_55_layer_call_and_return_conditional_losses_7040503

inputs0
matmul_readvariableop_resource:
identity��MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:���������2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
~
*__inference_dense_58_layer_call_fn_7042405

inputs
unknown:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_58_layer_call_and_return_conditional_losses_70405662
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_dense_57_layer_call_and_return_conditional_losses_7042398

inputs0
matmul_readvariableop_resource:
identity��MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:���������2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�,
�
S__inference_batch_normalization_66_layer_call_and_return_conditional_losses_7042028

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
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

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������2
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

:*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze�
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
:*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2
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
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_69_layer_call_fn_7042218

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_69_layer_call_and_return_conditional_losses_70402342
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_70_layer_call_fn_7042300

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_70_layer_call_and_return_conditional_losses_70404002
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_70_layer_call_fn_7042287

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_70_layer_call_and_return_conditional_losses_70403382
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_67_layer_call_and_return_conditional_losses_7039840

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�M
�
 __inference__traced_save_7042532
file_prefixe
asavev2_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_66_gamma_read_readvariableopd
`savev2_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_66_beta_read_readvariableope
asavev2_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_67_gamma_read_readvariableopd
`savev2_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_67_beta_read_readvariableope
asavev2_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_68_gamma_read_readvariableopd
`savev2_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_68_beta_read_readvariableope
asavev2_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_69_gamma_read_readvariableopd
`savev2_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_69_beta_read_readvariableope
asavev2_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_70_gamma_read_readvariableopd
`savev2_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_70_beta_read_readvariableopk
gsavev2_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_66_moving_mean_read_readvariableopo
ksavev2_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_66_moving_variance_read_readvariableopk
gsavev2_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_67_moving_mean_read_readvariableopo
ksavev2_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_67_moving_variance_read_readvariableopk
gsavev2_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_68_moving_mean_read_readvariableopo
ksavev2_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_68_moving_variance_read_readvariableopk
gsavev2_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_69_moving_mean_read_readvariableopo
ksavev2_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_69_moving_variance_read_readvariableopk
gsavev2_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_70_moving_mean_read_readvariableopo
ksavev2_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_70_moving_variance_read_readvariableopX
Tsavev2_nonshared_model_1_feed_forward_sub_net_11_dense_55_kernel_read_readvariableopX
Tsavev2_nonshared_model_1_feed_forward_sub_net_11_dense_56_kernel_read_readvariableopX
Tsavev2_nonshared_model_1_feed_forward_sub_net_11_dense_57_kernel_read_readvariableopX
Tsavev2_nonshared_model_1_feed_forward_sub_net_11_dense_58_kernel_read_readvariableopX
Tsavev2_nonshared_model_1_feed_forward_sub_net_11_dense_59_kernel_read_readvariableopV
Rsavev2_nonshared_model_1_feed_forward_sub_net_11_dense_59_bias_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0asavev2_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_66_gamma_read_readvariableop`savev2_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_66_beta_read_readvariableopasavev2_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_67_gamma_read_readvariableop`savev2_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_67_beta_read_readvariableopasavev2_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_68_gamma_read_readvariableop`savev2_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_68_beta_read_readvariableopasavev2_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_69_gamma_read_readvariableop`savev2_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_69_beta_read_readvariableopasavev2_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_70_gamma_read_readvariableop`savev2_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_70_beta_read_readvariableopgsavev2_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_66_moving_mean_read_readvariableopksavev2_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_66_moving_variance_read_readvariableopgsavev2_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_67_moving_mean_read_readvariableopksavev2_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_67_moving_variance_read_readvariableopgsavev2_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_68_moving_mean_read_readvariableopksavev2_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_68_moving_variance_read_readvariableopgsavev2_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_69_moving_mean_read_readvariableopksavev2_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_69_moving_variance_read_readvariableopgsavev2_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_70_moving_mean_read_readvariableopksavev2_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_70_moving_variance_read_readvariableopTsavev2_nonshared_model_1_feed_forward_sub_net_11_dense_55_kernel_read_readvariableopTsavev2_nonshared_model_1_feed_forward_sub_net_11_dense_56_kernel_read_readvariableopTsavev2_nonshared_model_1_feed_forward_sub_net_11_dense_57_kernel_read_readvariableopTsavev2_nonshared_model_1_feed_forward_sub_net_11_dense_58_kernel_read_readvariableopTsavev2_nonshared_model_1_feed_forward_sub_net_11_dense_59_kernel_read_readvariableopRsavev2_nonshared_model_1_feed_forward_sub_net_11_dense_59_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
�: ::::::::::::::::::::::::::: 2(
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
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
�
�
S__inference_batch_normalization_66_layer_call_and_return_conditional_losses_7041992

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�,
�
S__inference_batch_normalization_70_layer_call_and_return_conditional_losses_7042356

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
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

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������2
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

:*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze�
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
:*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2
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
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_dense_55_layer_call_and_return_conditional_losses_7042370

inputs0
matmul_readvariableop_resource:
identity��MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:���������2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
T__inference_feed_forward_sub_net_11_layer_call_and_return_conditional_losses_7041946
input_1L
>batch_normalization_66_assignmovingavg_readvariableop_resource:N
@batch_normalization_66_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_66_batchnorm_mul_readvariableop_resource:F
8batch_normalization_66_batchnorm_readvariableop_resource:9
'dense_55_matmul_readvariableop_resource:L
>batch_normalization_67_assignmovingavg_readvariableop_resource:N
@batch_normalization_67_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_67_batchnorm_mul_readvariableop_resource:F
8batch_normalization_67_batchnorm_readvariableop_resource:9
'dense_56_matmul_readvariableop_resource:L
>batch_normalization_68_assignmovingavg_readvariableop_resource:N
@batch_normalization_68_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_68_batchnorm_mul_readvariableop_resource:F
8batch_normalization_68_batchnorm_readvariableop_resource:9
'dense_57_matmul_readvariableop_resource:L
>batch_normalization_69_assignmovingavg_readvariableop_resource:N
@batch_normalization_69_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_69_batchnorm_mul_readvariableop_resource:F
8batch_normalization_69_batchnorm_readvariableop_resource:9
'dense_58_matmul_readvariableop_resource:L
>batch_normalization_70_assignmovingavg_readvariableop_resource:N
@batch_normalization_70_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_70_batchnorm_mul_readvariableop_resource:F
8batch_normalization_70_batchnorm_readvariableop_resource:9
'dense_59_matmul_readvariableop_resource:6
(dense_59_biasadd_readvariableop_resource:
identity��&batch_normalization_66/AssignMovingAvg�5batch_normalization_66/AssignMovingAvg/ReadVariableOp�(batch_normalization_66/AssignMovingAvg_1�7batch_normalization_66/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_66/batchnorm/ReadVariableOp�3batch_normalization_66/batchnorm/mul/ReadVariableOp�&batch_normalization_67/AssignMovingAvg�5batch_normalization_67/AssignMovingAvg/ReadVariableOp�(batch_normalization_67/AssignMovingAvg_1�7batch_normalization_67/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_67/batchnorm/ReadVariableOp�3batch_normalization_67/batchnorm/mul/ReadVariableOp�&batch_normalization_68/AssignMovingAvg�5batch_normalization_68/AssignMovingAvg/ReadVariableOp�(batch_normalization_68/AssignMovingAvg_1�7batch_normalization_68/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_68/batchnorm/ReadVariableOp�3batch_normalization_68/batchnorm/mul/ReadVariableOp�&batch_normalization_69/AssignMovingAvg�5batch_normalization_69/AssignMovingAvg/ReadVariableOp�(batch_normalization_69/AssignMovingAvg_1�7batch_normalization_69/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_69/batchnorm/ReadVariableOp�3batch_normalization_69/batchnorm/mul/ReadVariableOp�&batch_normalization_70/AssignMovingAvg�5batch_normalization_70/AssignMovingAvg/ReadVariableOp�(batch_normalization_70/AssignMovingAvg_1�7batch_normalization_70/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_70/batchnorm/ReadVariableOp�3batch_normalization_70/batchnorm/mul/ReadVariableOp�dense_55/MatMul/ReadVariableOp�dense_56/MatMul/ReadVariableOp�dense_57/MatMul/ReadVariableOp�dense_58/MatMul/ReadVariableOp�dense_59/BiasAdd/ReadVariableOp�dense_59/MatMul/ReadVariableOp�
5batch_normalization_66/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_66/moments/mean/reduction_indices�
#batch_normalization_66/moments/meanMeaninput_1>batch_normalization_66/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_66/moments/mean�
+batch_normalization_66/moments/StopGradientStopGradient,batch_normalization_66/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_66/moments/StopGradient�
0batch_normalization_66/moments/SquaredDifferenceSquaredDifferenceinput_14batch_normalization_66/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������22
0batch_normalization_66/moments/SquaredDifference�
9batch_normalization_66/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_66/moments/variance/reduction_indices�
'batch_normalization_66/moments/varianceMean4batch_normalization_66/moments/SquaredDifference:z:0Bbatch_normalization_66/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_66/moments/variance�
&batch_normalization_66/moments/SqueezeSqueeze,batch_normalization_66/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_66/moments/Squeeze�
(batch_normalization_66/moments/Squeeze_1Squeeze0batch_normalization_66/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_66/moments/Squeeze_1�
,batch_normalization_66/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_66/AssignMovingAvg/decay�
+batch_normalization_66/AssignMovingAvg/CastCast5batch_normalization_66/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_66/AssignMovingAvg/Cast�
5batch_normalization_66/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_66_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_66/AssignMovingAvg/ReadVariableOp�
*batch_normalization_66/AssignMovingAvg/subSub=batch_normalization_66/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_66/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_66/AssignMovingAvg/sub�
*batch_normalization_66/AssignMovingAvg/mulMul.batch_normalization_66/AssignMovingAvg/sub:z:0/batch_normalization_66/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_66/AssignMovingAvg/mul�
&batch_normalization_66/AssignMovingAvgAssignSubVariableOp>batch_normalization_66_assignmovingavg_readvariableop_resource.batch_normalization_66/AssignMovingAvg/mul:z:06^batch_normalization_66/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_66/AssignMovingAvg�
.batch_normalization_66/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_66/AssignMovingAvg_1/decay�
-batch_normalization_66/AssignMovingAvg_1/CastCast7batch_normalization_66/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_66/AssignMovingAvg_1/Cast�
7batch_normalization_66/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_66_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_66/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_66/AssignMovingAvg_1/subSub?batch_normalization_66/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_66/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_66/AssignMovingAvg_1/sub�
,batch_normalization_66/AssignMovingAvg_1/mulMul0batch_normalization_66/AssignMovingAvg_1/sub:z:01batch_normalization_66/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_66/AssignMovingAvg_1/mul�
(batch_normalization_66/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_66_assignmovingavg_1_readvariableop_resource0batch_normalization_66/AssignMovingAvg_1/mul:z:08^batch_normalization_66/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_66/AssignMovingAvg_1�
&batch_normalization_66/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_66/batchnorm/add/y�
$batch_normalization_66/batchnorm/addAddV21batch_normalization_66/moments/Squeeze_1:output:0/batch_normalization_66/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_66/batchnorm/add�
&batch_normalization_66/batchnorm/RsqrtRsqrt(batch_normalization_66/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_66/batchnorm/Rsqrt�
3batch_normalization_66/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_66_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_66/batchnorm/mul/ReadVariableOp�
$batch_normalization_66/batchnorm/mulMul*batch_normalization_66/batchnorm/Rsqrt:y:0;batch_normalization_66/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_66/batchnorm/mul�
&batch_normalization_66/batchnorm/mul_1Mulinput_1(batch_normalization_66/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_66/batchnorm/mul_1�
&batch_normalization_66/batchnorm/mul_2Mul/batch_normalization_66/moments/Squeeze:output:0(batch_normalization_66/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_66/batchnorm/mul_2�
/batch_normalization_66/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_66_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_66/batchnorm/ReadVariableOp�
$batch_normalization_66/batchnorm/subSub7batch_normalization_66/batchnorm/ReadVariableOp:value:0*batch_normalization_66/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_66/batchnorm/sub�
&batch_normalization_66/batchnorm/add_1AddV2*batch_normalization_66/batchnorm/mul_1:z:0(batch_normalization_66/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_66/batchnorm/add_1�
dense_55/MatMul/ReadVariableOpReadVariableOp'dense_55_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_55/MatMul/ReadVariableOp�
dense_55/MatMulMatMul*batch_normalization_66/batchnorm/add_1:z:0&dense_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_55/MatMul�
5batch_normalization_67/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_67/moments/mean/reduction_indices�
#batch_normalization_67/moments/meanMeandense_55/MatMul:product:0>batch_normalization_67/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_67/moments/mean�
+batch_normalization_67/moments/StopGradientStopGradient,batch_normalization_67/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_67/moments/StopGradient�
0batch_normalization_67/moments/SquaredDifferenceSquaredDifferencedense_55/MatMul:product:04batch_normalization_67/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������22
0batch_normalization_67/moments/SquaredDifference�
9batch_normalization_67/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_67/moments/variance/reduction_indices�
'batch_normalization_67/moments/varianceMean4batch_normalization_67/moments/SquaredDifference:z:0Bbatch_normalization_67/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_67/moments/variance�
&batch_normalization_67/moments/SqueezeSqueeze,batch_normalization_67/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_67/moments/Squeeze�
(batch_normalization_67/moments/Squeeze_1Squeeze0batch_normalization_67/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_67/moments/Squeeze_1�
,batch_normalization_67/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_67/AssignMovingAvg/decay�
+batch_normalization_67/AssignMovingAvg/CastCast5batch_normalization_67/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_67/AssignMovingAvg/Cast�
5batch_normalization_67/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_67_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_67/AssignMovingAvg/ReadVariableOp�
*batch_normalization_67/AssignMovingAvg/subSub=batch_normalization_67/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_67/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_67/AssignMovingAvg/sub�
*batch_normalization_67/AssignMovingAvg/mulMul.batch_normalization_67/AssignMovingAvg/sub:z:0/batch_normalization_67/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_67/AssignMovingAvg/mul�
&batch_normalization_67/AssignMovingAvgAssignSubVariableOp>batch_normalization_67_assignmovingavg_readvariableop_resource.batch_normalization_67/AssignMovingAvg/mul:z:06^batch_normalization_67/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_67/AssignMovingAvg�
.batch_normalization_67/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_67/AssignMovingAvg_1/decay�
-batch_normalization_67/AssignMovingAvg_1/CastCast7batch_normalization_67/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_67/AssignMovingAvg_1/Cast�
7batch_normalization_67/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_67_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_67/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_67/AssignMovingAvg_1/subSub?batch_normalization_67/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_67/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_67/AssignMovingAvg_1/sub�
,batch_normalization_67/AssignMovingAvg_1/mulMul0batch_normalization_67/AssignMovingAvg_1/sub:z:01batch_normalization_67/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_67/AssignMovingAvg_1/mul�
(batch_normalization_67/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_67_assignmovingavg_1_readvariableop_resource0batch_normalization_67/AssignMovingAvg_1/mul:z:08^batch_normalization_67/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_67/AssignMovingAvg_1�
&batch_normalization_67/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_67/batchnorm/add/y�
$batch_normalization_67/batchnorm/addAddV21batch_normalization_67/moments/Squeeze_1:output:0/batch_normalization_67/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_67/batchnorm/add�
&batch_normalization_67/batchnorm/RsqrtRsqrt(batch_normalization_67/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_67/batchnorm/Rsqrt�
3batch_normalization_67/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_67_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_67/batchnorm/mul/ReadVariableOp�
$batch_normalization_67/batchnorm/mulMul*batch_normalization_67/batchnorm/Rsqrt:y:0;batch_normalization_67/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_67/batchnorm/mul�
&batch_normalization_67/batchnorm/mul_1Muldense_55/MatMul:product:0(batch_normalization_67/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_67/batchnorm/mul_1�
&batch_normalization_67/batchnorm/mul_2Mul/batch_normalization_67/moments/Squeeze:output:0(batch_normalization_67/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_67/batchnorm/mul_2�
/batch_normalization_67/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_67_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_67/batchnorm/ReadVariableOp�
$batch_normalization_67/batchnorm/subSub7batch_normalization_67/batchnorm/ReadVariableOp:value:0*batch_normalization_67/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_67/batchnorm/sub�
&batch_normalization_67/batchnorm/add_1AddV2*batch_normalization_67/batchnorm/mul_1:z:0(batch_normalization_67/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_67/batchnorm/add_1r
ReluRelu*batch_normalization_67/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu�
dense_56/MatMul/ReadVariableOpReadVariableOp'dense_56_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_56/MatMul/ReadVariableOp�
dense_56/MatMulMatMulRelu:activations:0&dense_56/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_56/MatMul�
5batch_normalization_68/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_68/moments/mean/reduction_indices�
#batch_normalization_68/moments/meanMeandense_56/MatMul:product:0>batch_normalization_68/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_68/moments/mean�
+batch_normalization_68/moments/StopGradientStopGradient,batch_normalization_68/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_68/moments/StopGradient�
0batch_normalization_68/moments/SquaredDifferenceSquaredDifferencedense_56/MatMul:product:04batch_normalization_68/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������22
0batch_normalization_68/moments/SquaredDifference�
9batch_normalization_68/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_68/moments/variance/reduction_indices�
'batch_normalization_68/moments/varianceMean4batch_normalization_68/moments/SquaredDifference:z:0Bbatch_normalization_68/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_68/moments/variance�
&batch_normalization_68/moments/SqueezeSqueeze,batch_normalization_68/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_68/moments/Squeeze�
(batch_normalization_68/moments/Squeeze_1Squeeze0batch_normalization_68/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_68/moments/Squeeze_1�
,batch_normalization_68/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_68/AssignMovingAvg/decay�
+batch_normalization_68/AssignMovingAvg/CastCast5batch_normalization_68/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_68/AssignMovingAvg/Cast�
5batch_normalization_68/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_68_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_68/AssignMovingAvg/ReadVariableOp�
*batch_normalization_68/AssignMovingAvg/subSub=batch_normalization_68/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_68/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_68/AssignMovingAvg/sub�
*batch_normalization_68/AssignMovingAvg/mulMul.batch_normalization_68/AssignMovingAvg/sub:z:0/batch_normalization_68/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_68/AssignMovingAvg/mul�
&batch_normalization_68/AssignMovingAvgAssignSubVariableOp>batch_normalization_68_assignmovingavg_readvariableop_resource.batch_normalization_68/AssignMovingAvg/mul:z:06^batch_normalization_68/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_68/AssignMovingAvg�
.batch_normalization_68/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_68/AssignMovingAvg_1/decay�
-batch_normalization_68/AssignMovingAvg_1/CastCast7batch_normalization_68/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_68/AssignMovingAvg_1/Cast�
7batch_normalization_68/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_68_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_68/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_68/AssignMovingAvg_1/subSub?batch_normalization_68/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_68/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_68/AssignMovingAvg_1/sub�
,batch_normalization_68/AssignMovingAvg_1/mulMul0batch_normalization_68/AssignMovingAvg_1/sub:z:01batch_normalization_68/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_68/AssignMovingAvg_1/mul�
(batch_normalization_68/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_68_assignmovingavg_1_readvariableop_resource0batch_normalization_68/AssignMovingAvg_1/mul:z:08^batch_normalization_68/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_68/AssignMovingAvg_1�
&batch_normalization_68/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_68/batchnorm/add/y�
$batch_normalization_68/batchnorm/addAddV21batch_normalization_68/moments/Squeeze_1:output:0/batch_normalization_68/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_68/batchnorm/add�
&batch_normalization_68/batchnorm/RsqrtRsqrt(batch_normalization_68/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_68/batchnorm/Rsqrt�
3batch_normalization_68/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_68_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_68/batchnorm/mul/ReadVariableOp�
$batch_normalization_68/batchnorm/mulMul*batch_normalization_68/batchnorm/Rsqrt:y:0;batch_normalization_68/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_68/batchnorm/mul�
&batch_normalization_68/batchnorm/mul_1Muldense_56/MatMul:product:0(batch_normalization_68/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_68/batchnorm/mul_1�
&batch_normalization_68/batchnorm/mul_2Mul/batch_normalization_68/moments/Squeeze:output:0(batch_normalization_68/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_68/batchnorm/mul_2�
/batch_normalization_68/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_68_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_68/batchnorm/ReadVariableOp�
$batch_normalization_68/batchnorm/subSub7batch_normalization_68/batchnorm/ReadVariableOp:value:0*batch_normalization_68/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_68/batchnorm/sub�
&batch_normalization_68/batchnorm/add_1AddV2*batch_normalization_68/batchnorm/mul_1:z:0(batch_normalization_68/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_68/batchnorm/add_1v
Relu_1Relu*batch_normalization_68/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_1�
dense_57/MatMul/ReadVariableOpReadVariableOp'dense_57_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_57/MatMul/ReadVariableOp�
dense_57/MatMulMatMulRelu_1:activations:0&dense_57/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_57/MatMul�
5batch_normalization_69/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_69/moments/mean/reduction_indices�
#batch_normalization_69/moments/meanMeandense_57/MatMul:product:0>batch_normalization_69/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_69/moments/mean�
+batch_normalization_69/moments/StopGradientStopGradient,batch_normalization_69/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_69/moments/StopGradient�
0batch_normalization_69/moments/SquaredDifferenceSquaredDifferencedense_57/MatMul:product:04batch_normalization_69/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������22
0batch_normalization_69/moments/SquaredDifference�
9batch_normalization_69/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_69/moments/variance/reduction_indices�
'batch_normalization_69/moments/varianceMean4batch_normalization_69/moments/SquaredDifference:z:0Bbatch_normalization_69/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_69/moments/variance�
&batch_normalization_69/moments/SqueezeSqueeze,batch_normalization_69/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_69/moments/Squeeze�
(batch_normalization_69/moments/Squeeze_1Squeeze0batch_normalization_69/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_69/moments/Squeeze_1�
,batch_normalization_69/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_69/AssignMovingAvg/decay�
+batch_normalization_69/AssignMovingAvg/CastCast5batch_normalization_69/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_69/AssignMovingAvg/Cast�
5batch_normalization_69/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_69_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_69/AssignMovingAvg/ReadVariableOp�
*batch_normalization_69/AssignMovingAvg/subSub=batch_normalization_69/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_69/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_69/AssignMovingAvg/sub�
*batch_normalization_69/AssignMovingAvg/mulMul.batch_normalization_69/AssignMovingAvg/sub:z:0/batch_normalization_69/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_69/AssignMovingAvg/mul�
&batch_normalization_69/AssignMovingAvgAssignSubVariableOp>batch_normalization_69_assignmovingavg_readvariableop_resource.batch_normalization_69/AssignMovingAvg/mul:z:06^batch_normalization_69/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_69/AssignMovingAvg�
.batch_normalization_69/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_69/AssignMovingAvg_1/decay�
-batch_normalization_69/AssignMovingAvg_1/CastCast7batch_normalization_69/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_69/AssignMovingAvg_1/Cast�
7batch_normalization_69/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_69_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_69/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_69/AssignMovingAvg_1/subSub?batch_normalization_69/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_69/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_69/AssignMovingAvg_1/sub�
,batch_normalization_69/AssignMovingAvg_1/mulMul0batch_normalization_69/AssignMovingAvg_1/sub:z:01batch_normalization_69/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_69/AssignMovingAvg_1/mul�
(batch_normalization_69/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_69_assignmovingavg_1_readvariableop_resource0batch_normalization_69/AssignMovingAvg_1/mul:z:08^batch_normalization_69/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_69/AssignMovingAvg_1�
&batch_normalization_69/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_69/batchnorm/add/y�
$batch_normalization_69/batchnorm/addAddV21batch_normalization_69/moments/Squeeze_1:output:0/batch_normalization_69/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_69/batchnorm/add�
&batch_normalization_69/batchnorm/RsqrtRsqrt(batch_normalization_69/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_69/batchnorm/Rsqrt�
3batch_normalization_69/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_69_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_69/batchnorm/mul/ReadVariableOp�
$batch_normalization_69/batchnorm/mulMul*batch_normalization_69/batchnorm/Rsqrt:y:0;batch_normalization_69/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_69/batchnorm/mul�
&batch_normalization_69/batchnorm/mul_1Muldense_57/MatMul:product:0(batch_normalization_69/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_69/batchnorm/mul_1�
&batch_normalization_69/batchnorm/mul_2Mul/batch_normalization_69/moments/Squeeze:output:0(batch_normalization_69/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_69/batchnorm/mul_2�
/batch_normalization_69/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_69_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_69/batchnorm/ReadVariableOp�
$batch_normalization_69/batchnorm/subSub7batch_normalization_69/batchnorm/ReadVariableOp:value:0*batch_normalization_69/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_69/batchnorm/sub�
&batch_normalization_69/batchnorm/add_1AddV2*batch_normalization_69/batchnorm/mul_1:z:0(batch_normalization_69/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_69/batchnorm/add_1v
Relu_2Relu*batch_normalization_69/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_2�
dense_58/MatMul/ReadVariableOpReadVariableOp'dense_58_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_58/MatMul/ReadVariableOp�
dense_58/MatMulMatMulRelu_2:activations:0&dense_58/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_58/MatMul�
5batch_normalization_70/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_70/moments/mean/reduction_indices�
#batch_normalization_70/moments/meanMeandense_58/MatMul:product:0>batch_normalization_70/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_70/moments/mean�
+batch_normalization_70/moments/StopGradientStopGradient,batch_normalization_70/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_70/moments/StopGradient�
0batch_normalization_70/moments/SquaredDifferenceSquaredDifferencedense_58/MatMul:product:04batch_normalization_70/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������22
0batch_normalization_70/moments/SquaredDifference�
9batch_normalization_70/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_70/moments/variance/reduction_indices�
'batch_normalization_70/moments/varianceMean4batch_normalization_70/moments/SquaredDifference:z:0Bbatch_normalization_70/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_70/moments/variance�
&batch_normalization_70/moments/SqueezeSqueeze,batch_normalization_70/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_70/moments/Squeeze�
(batch_normalization_70/moments/Squeeze_1Squeeze0batch_normalization_70/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_70/moments/Squeeze_1�
,batch_normalization_70/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_70/AssignMovingAvg/decay�
+batch_normalization_70/AssignMovingAvg/CastCast5batch_normalization_70/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_70/AssignMovingAvg/Cast�
5batch_normalization_70/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_70_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_70/AssignMovingAvg/ReadVariableOp�
*batch_normalization_70/AssignMovingAvg/subSub=batch_normalization_70/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_70/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_70/AssignMovingAvg/sub�
*batch_normalization_70/AssignMovingAvg/mulMul.batch_normalization_70/AssignMovingAvg/sub:z:0/batch_normalization_70/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_70/AssignMovingAvg/mul�
&batch_normalization_70/AssignMovingAvgAssignSubVariableOp>batch_normalization_70_assignmovingavg_readvariableop_resource.batch_normalization_70/AssignMovingAvg/mul:z:06^batch_normalization_70/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_70/AssignMovingAvg�
.batch_normalization_70/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_70/AssignMovingAvg_1/decay�
-batch_normalization_70/AssignMovingAvg_1/CastCast7batch_normalization_70/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_70/AssignMovingAvg_1/Cast�
7batch_normalization_70/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_70_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_70/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_70/AssignMovingAvg_1/subSub?batch_normalization_70/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_70/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_70/AssignMovingAvg_1/sub�
,batch_normalization_70/AssignMovingAvg_1/mulMul0batch_normalization_70/AssignMovingAvg_1/sub:z:01batch_normalization_70/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_70/AssignMovingAvg_1/mul�
(batch_normalization_70/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_70_assignmovingavg_1_readvariableop_resource0batch_normalization_70/AssignMovingAvg_1/mul:z:08^batch_normalization_70/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_70/AssignMovingAvg_1�
&batch_normalization_70/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_70/batchnorm/add/y�
$batch_normalization_70/batchnorm/addAddV21batch_normalization_70/moments/Squeeze_1:output:0/batch_normalization_70/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_70/batchnorm/add�
&batch_normalization_70/batchnorm/RsqrtRsqrt(batch_normalization_70/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_70/batchnorm/Rsqrt�
3batch_normalization_70/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_70_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_70/batchnorm/mul/ReadVariableOp�
$batch_normalization_70/batchnorm/mulMul*batch_normalization_70/batchnorm/Rsqrt:y:0;batch_normalization_70/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_70/batchnorm/mul�
&batch_normalization_70/batchnorm/mul_1Muldense_58/MatMul:product:0(batch_normalization_70/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_70/batchnorm/mul_1�
&batch_normalization_70/batchnorm/mul_2Mul/batch_normalization_70/moments/Squeeze:output:0(batch_normalization_70/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_70/batchnorm/mul_2�
/batch_normalization_70/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_70_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_70/batchnorm/ReadVariableOp�
$batch_normalization_70/batchnorm/subSub7batch_normalization_70/batchnorm/ReadVariableOp:value:0*batch_normalization_70/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_70/batchnorm/sub�
&batch_normalization_70/batchnorm/add_1AddV2*batch_normalization_70/batchnorm/mul_1:z:0(batch_normalization_70/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_70/batchnorm/add_1v
Relu_3Relu*batch_normalization_70/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_3�
dense_59/MatMul/ReadVariableOpReadVariableOp'dense_59_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_59/MatMul/ReadVariableOp�
dense_59/MatMulMatMulRelu_3:activations:0&dense_59/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_59/MatMul�
dense_59/BiasAdd/ReadVariableOpReadVariableOp(dense_59_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_59/BiasAdd/ReadVariableOp�
dense_59/BiasAddBiasAdddense_59/MatMul:product:0'dense_59/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_59/BiasAddt
IdentityIdentitydense_59/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp'^batch_normalization_66/AssignMovingAvg6^batch_normalization_66/AssignMovingAvg/ReadVariableOp)^batch_normalization_66/AssignMovingAvg_18^batch_normalization_66/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_66/batchnorm/ReadVariableOp4^batch_normalization_66/batchnorm/mul/ReadVariableOp'^batch_normalization_67/AssignMovingAvg6^batch_normalization_67/AssignMovingAvg/ReadVariableOp)^batch_normalization_67/AssignMovingAvg_18^batch_normalization_67/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_67/batchnorm/ReadVariableOp4^batch_normalization_67/batchnorm/mul/ReadVariableOp'^batch_normalization_68/AssignMovingAvg6^batch_normalization_68/AssignMovingAvg/ReadVariableOp)^batch_normalization_68/AssignMovingAvg_18^batch_normalization_68/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_68/batchnorm/ReadVariableOp4^batch_normalization_68/batchnorm/mul/ReadVariableOp'^batch_normalization_69/AssignMovingAvg6^batch_normalization_69/AssignMovingAvg/ReadVariableOp)^batch_normalization_69/AssignMovingAvg_18^batch_normalization_69/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_69/batchnorm/ReadVariableOp4^batch_normalization_69/batchnorm/mul/ReadVariableOp'^batch_normalization_70/AssignMovingAvg6^batch_normalization_70/AssignMovingAvg/ReadVariableOp)^batch_normalization_70/AssignMovingAvg_18^batch_normalization_70/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_70/batchnorm/ReadVariableOp4^batch_normalization_70/batchnorm/mul/ReadVariableOp^dense_55/MatMul/ReadVariableOp^dense_56/MatMul/ReadVariableOp^dense_57/MatMul/ReadVariableOp^dense_58/MatMul/ReadVariableOp ^dense_59/BiasAdd/ReadVariableOp^dense_59/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������: : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_66/AssignMovingAvg&batch_normalization_66/AssignMovingAvg2n
5batch_normalization_66/AssignMovingAvg/ReadVariableOp5batch_normalization_66/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_66/AssignMovingAvg_1(batch_normalization_66/AssignMovingAvg_12r
7batch_normalization_66/AssignMovingAvg_1/ReadVariableOp7batch_normalization_66/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_66/batchnorm/ReadVariableOp/batch_normalization_66/batchnorm/ReadVariableOp2j
3batch_normalization_66/batchnorm/mul/ReadVariableOp3batch_normalization_66/batchnorm/mul/ReadVariableOp2P
&batch_normalization_67/AssignMovingAvg&batch_normalization_67/AssignMovingAvg2n
5batch_normalization_67/AssignMovingAvg/ReadVariableOp5batch_normalization_67/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_67/AssignMovingAvg_1(batch_normalization_67/AssignMovingAvg_12r
7batch_normalization_67/AssignMovingAvg_1/ReadVariableOp7batch_normalization_67/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_67/batchnorm/ReadVariableOp/batch_normalization_67/batchnorm/ReadVariableOp2j
3batch_normalization_67/batchnorm/mul/ReadVariableOp3batch_normalization_67/batchnorm/mul/ReadVariableOp2P
&batch_normalization_68/AssignMovingAvg&batch_normalization_68/AssignMovingAvg2n
5batch_normalization_68/AssignMovingAvg/ReadVariableOp5batch_normalization_68/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_68/AssignMovingAvg_1(batch_normalization_68/AssignMovingAvg_12r
7batch_normalization_68/AssignMovingAvg_1/ReadVariableOp7batch_normalization_68/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_68/batchnorm/ReadVariableOp/batch_normalization_68/batchnorm/ReadVariableOp2j
3batch_normalization_68/batchnorm/mul/ReadVariableOp3batch_normalization_68/batchnorm/mul/ReadVariableOp2P
&batch_normalization_69/AssignMovingAvg&batch_normalization_69/AssignMovingAvg2n
5batch_normalization_69/AssignMovingAvg/ReadVariableOp5batch_normalization_69/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_69/AssignMovingAvg_1(batch_normalization_69/AssignMovingAvg_12r
7batch_normalization_69/AssignMovingAvg_1/ReadVariableOp7batch_normalization_69/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_69/batchnorm/ReadVariableOp/batch_normalization_69/batchnorm/ReadVariableOp2j
3batch_normalization_69/batchnorm/mul/ReadVariableOp3batch_normalization_69/batchnorm/mul/ReadVariableOp2P
&batch_normalization_70/AssignMovingAvg&batch_normalization_70/AssignMovingAvg2n
5batch_normalization_70/AssignMovingAvg/ReadVariableOp5batch_normalization_70/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_70/AssignMovingAvg_1(batch_normalization_70/AssignMovingAvg_12r
7batch_normalization_70/AssignMovingAvg_1/ReadVariableOp7batch_normalization_70/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_70/batchnorm/ReadVariableOp/batch_normalization_70/batchnorm/ReadVariableOp2j
3batch_normalization_70/batchnorm/mul/ReadVariableOp3batch_normalization_70/batchnorm/mul/ReadVariableOp2@
dense_55/MatMul/ReadVariableOpdense_55/MatMul/ReadVariableOp2@
dense_56/MatMul/ReadVariableOpdense_56/MatMul/ReadVariableOp2@
dense_57/MatMul/ReadVariableOpdense_57/MatMul/ReadVariableOp2@
dense_58/MatMul/ReadVariableOpdense_58/MatMul/ReadVariableOp2B
dense_59/BiasAdd/ReadVariableOpdense_59/BiasAdd/ReadVariableOp2@
dense_59/MatMul/ReadVariableOpdense_59/MatMul/ReadVariableOp:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�,
�
S__inference_batch_normalization_68_layer_call_and_return_conditional_losses_7040068

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
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

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������2
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

:*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze�
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
:*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2
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
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
T__inference_feed_forward_sub_net_11_layer_call_and_return_conditional_losses_7041468
xF
8batch_normalization_66_batchnorm_readvariableop_resource:J
<batch_normalization_66_batchnorm_mul_readvariableop_resource:H
:batch_normalization_66_batchnorm_readvariableop_1_resource:H
:batch_normalization_66_batchnorm_readvariableop_2_resource:9
'dense_55_matmul_readvariableop_resource:F
8batch_normalization_67_batchnorm_readvariableop_resource:J
<batch_normalization_67_batchnorm_mul_readvariableop_resource:H
:batch_normalization_67_batchnorm_readvariableop_1_resource:H
:batch_normalization_67_batchnorm_readvariableop_2_resource:9
'dense_56_matmul_readvariableop_resource:F
8batch_normalization_68_batchnorm_readvariableop_resource:J
<batch_normalization_68_batchnorm_mul_readvariableop_resource:H
:batch_normalization_68_batchnorm_readvariableop_1_resource:H
:batch_normalization_68_batchnorm_readvariableop_2_resource:9
'dense_57_matmul_readvariableop_resource:F
8batch_normalization_69_batchnorm_readvariableop_resource:J
<batch_normalization_69_batchnorm_mul_readvariableop_resource:H
:batch_normalization_69_batchnorm_readvariableop_1_resource:H
:batch_normalization_69_batchnorm_readvariableop_2_resource:9
'dense_58_matmul_readvariableop_resource:F
8batch_normalization_70_batchnorm_readvariableop_resource:J
<batch_normalization_70_batchnorm_mul_readvariableop_resource:H
:batch_normalization_70_batchnorm_readvariableop_1_resource:H
:batch_normalization_70_batchnorm_readvariableop_2_resource:9
'dense_59_matmul_readvariableop_resource:6
(dense_59_biasadd_readvariableop_resource:
identity��/batch_normalization_66/batchnorm/ReadVariableOp�1batch_normalization_66/batchnorm/ReadVariableOp_1�1batch_normalization_66/batchnorm/ReadVariableOp_2�3batch_normalization_66/batchnorm/mul/ReadVariableOp�/batch_normalization_67/batchnorm/ReadVariableOp�1batch_normalization_67/batchnorm/ReadVariableOp_1�1batch_normalization_67/batchnorm/ReadVariableOp_2�3batch_normalization_67/batchnorm/mul/ReadVariableOp�/batch_normalization_68/batchnorm/ReadVariableOp�1batch_normalization_68/batchnorm/ReadVariableOp_1�1batch_normalization_68/batchnorm/ReadVariableOp_2�3batch_normalization_68/batchnorm/mul/ReadVariableOp�/batch_normalization_69/batchnorm/ReadVariableOp�1batch_normalization_69/batchnorm/ReadVariableOp_1�1batch_normalization_69/batchnorm/ReadVariableOp_2�3batch_normalization_69/batchnorm/mul/ReadVariableOp�/batch_normalization_70/batchnorm/ReadVariableOp�1batch_normalization_70/batchnorm/ReadVariableOp_1�1batch_normalization_70/batchnorm/ReadVariableOp_2�3batch_normalization_70/batchnorm/mul/ReadVariableOp�dense_55/MatMul/ReadVariableOp�dense_56/MatMul/ReadVariableOp�dense_57/MatMul/ReadVariableOp�dense_58/MatMul/ReadVariableOp�dense_59/BiasAdd/ReadVariableOp�dense_59/MatMul/ReadVariableOp�
/batch_normalization_66/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_66_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_66/batchnorm/ReadVariableOp�
&batch_normalization_66/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_66/batchnorm/add/y�
$batch_normalization_66/batchnorm/addAddV27batch_normalization_66/batchnorm/ReadVariableOp:value:0/batch_normalization_66/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_66/batchnorm/add�
&batch_normalization_66/batchnorm/RsqrtRsqrt(batch_normalization_66/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_66/batchnorm/Rsqrt�
3batch_normalization_66/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_66_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_66/batchnorm/mul/ReadVariableOp�
$batch_normalization_66/batchnorm/mulMul*batch_normalization_66/batchnorm/Rsqrt:y:0;batch_normalization_66/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_66/batchnorm/mul�
&batch_normalization_66/batchnorm/mul_1Mulx(batch_normalization_66/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_66/batchnorm/mul_1�
1batch_normalization_66/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_66_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_66/batchnorm/ReadVariableOp_1�
&batch_normalization_66/batchnorm/mul_2Mul9batch_normalization_66/batchnorm/ReadVariableOp_1:value:0(batch_normalization_66/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_66/batchnorm/mul_2�
1batch_normalization_66/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_66_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_66/batchnorm/ReadVariableOp_2�
$batch_normalization_66/batchnorm/subSub9batch_normalization_66/batchnorm/ReadVariableOp_2:value:0*batch_normalization_66/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_66/batchnorm/sub�
&batch_normalization_66/batchnorm/add_1AddV2*batch_normalization_66/batchnorm/mul_1:z:0(batch_normalization_66/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_66/batchnorm/add_1�
dense_55/MatMul/ReadVariableOpReadVariableOp'dense_55_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_55/MatMul/ReadVariableOp�
dense_55/MatMulMatMul*batch_normalization_66/batchnorm/add_1:z:0&dense_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_55/MatMul�
/batch_normalization_67/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_67_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_67/batchnorm/ReadVariableOp�
&batch_normalization_67/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_67/batchnorm/add/y�
$batch_normalization_67/batchnorm/addAddV27batch_normalization_67/batchnorm/ReadVariableOp:value:0/batch_normalization_67/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_67/batchnorm/add�
&batch_normalization_67/batchnorm/RsqrtRsqrt(batch_normalization_67/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_67/batchnorm/Rsqrt�
3batch_normalization_67/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_67_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_67/batchnorm/mul/ReadVariableOp�
$batch_normalization_67/batchnorm/mulMul*batch_normalization_67/batchnorm/Rsqrt:y:0;batch_normalization_67/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_67/batchnorm/mul�
&batch_normalization_67/batchnorm/mul_1Muldense_55/MatMul:product:0(batch_normalization_67/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_67/batchnorm/mul_1�
1batch_normalization_67/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_67_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_67/batchnorm/ReadVariableOp_1�
&batch_normalization_67/batchnorm/mul_2Mul9batch_normalization_67/batchnorm/ReadVariableOp_1:value:0(batch_normalization_67/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_67/batchnorm/mul_2�
1batch_normalization_67/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_67_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_67/batchnorm/ReadVariableOp_2�
$batch_normalization_67/batchnorm/subSub9batch_normalization_67/batchnorm/ReadVariableOp_2:value:0*batch_normalization_67/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_67/batchnorm/sub�
&batch_normalization_67/batchnorm/add_1AddV2*batch_normalization_67/batchnorm/mul_1:z:0(batch_normalization_67/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_67/batchnorm/add_1r
ReluRelu*batch_normalization_67/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu�
dense_56/MatMul/ReadVariableOpReadVariableOp'dense_56_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_56/MatMul/ReadVariableOp�
dense_56/MatMulMatMulRelu:activations:0&dense_56/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_56/MatMul�
/batch_normalization_68/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_68_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_68/batchnorm/ReadVariableOp�
&batch_normalization_68/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_68/batchnorm/add/y�
$batch_normalization_68/batchnorm/addAddV27batch_normalization_68/batchnorm/ReadVariableOp:value:0/batch_normalization_68/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_68/batchnorm/add�
&batch_normalization_68/batchnorm/RsqrtRsqrt(batch_normalization_68/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_68/batchnorm/Rsqrt�
3batch_normalization_68/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_68_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_68/batchnorm/mul/ReadVariableOp�
$batch_normalization_68/batchnorm/mulMul*batch_normalization_68/batchnorm/Rsqrt:y:0;batch_normalization_68/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_68/batchnorm/mul�
&batch_normalization_68/batchnorm/mul_1Muldense_56/MatMul:product:0(batch_normalization_68/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_68/batchnorm/mul_1�
1batch_normalization_68/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_68_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_68/batchnorm/ReadVariableOp_1�
&batch_normalization_68/batchnorm/mul_2Mul9batch_normalization_68/batchnorm/ReadVariableOp_1:value:0(batch_normalization_68/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_68/batchnorm/mul_2�
1batch_normalization_68/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_68_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_68/batchnorm/ReadVariableOp_2�
$batch_normalization_68/batchnorm/subSub9batch_normalization_68/batchnorm/ReadVariableOp_2:value:0*batch_normalization_68/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_68/batchnorm/sub�
&batch_normalization_68/batchnorm/add_1AddV2*batch_normalization_68/batchnorm/mul_1:z:0(batch_normalization_68/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_68/batchnorm/add_1v
Relu_1Relu*batch_normalization_68/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_1�
dense_57/MatMul/ReadVariableOpReadVariableOp'dense_57_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_57/MatMul/ReadVariableOp�
dense_57/MatMulMatMulRelu_1:activations:0&dense_57/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_57/MatMul�
/batch_normalization_69/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_69_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_69/batchnorm/ReadVariableOp�
&batch_normalization_69/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_69/batchnorm/add/y�
$batch_normalization_69/batchnorm/addAddV27batch_normalization_69/batchnorm/ReadVariableOp:value:0/batch_normalization_69/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_69/batchnorm/add�
&batch_normalization_69/batchnorm/RsqrtRsqrt(batch_normalization_69/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_69/batchnorm/Rsqrt�
3batch_normalization_69/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_69_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_69/batchnorm/mul/ReadVariableOp�
$batch_normalization_69/batchnorm/mulMul*batch_normalization_69/batchnorm/Rsqrt:y:0;batch_normalization_69/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_69/batchnorm/mul�
&batch_normalization_69/batchnorm/mul_1Muldense_57/MatMul:product:0(batch_normalization_69/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_69/batchnorm/mul_1�
1batch_normalization_69/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_69_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_69/batchnorm/ReadVariableOp_1�
&batch_normalization_69/batchnorm/mul_2Mul9batch_normalization_69/batchnorm/ReadVariableOp_1:value:0(batch_normalization_69/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_69/batchnorm/mul_2�
1batch_normalization_69/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_69_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_69/batchnorm/ReadVariableOp_2�
$batch_normalization_69/batchnorm/subSub9batch_normalization_69/batchnorm/ReadVariableOp_2:value:0*batch_normalization_69/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_69/batchnorm/sub�
&batch_normalization_69/batchnorm/add_1AddV2*batch_normalization_69/batchnorm/mul_1:z:0(batch_normalization_69/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_69/batchnorm/add_1v
Relu_2Relu*batch_normalization_69/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_2�
dense_58/MatMul/ReadVariableOpReadVariableOp'dense_58_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_58/MatMul/ReadVariableOp�
dense_58/MatMulMatMulRelu_2:activations:0&dense_58/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_58/MatMul�
/batch_normalization_70/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_70_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_70/batchnorm/ReadVariableOp�
&batch_normalization_70/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_70/batchnorm/add/y�
$batch_normalization_70/batchnorm/addAddV27batch_normalization_70/batchnorm/ReadVariableOp:value:0/batch_normalization_70/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_70/batchnorm/add�
&batch_normalization_70/batchnorm/RsqrtRsqrt(batch_normalization_70/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_70/batchnorm/Rsqrt�
3batch_normalization_70/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_70_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_70/batchnorm/mul/ReadVariableOp�
$batch_normalization_70/batchnorm/mulMul*batch_normalization_70/batchnorm/Rsqrt:y:0;batch_normalization_70/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_70/batchnorm/mul�
&batch_normalization_70/batchnorm/mul_1Muldense_58/MatMul:product:0(batch_normalization_70/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_70/batchnorm/mul_1�
1batch_normalization_70/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_70_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_70/batchnorm/ReadVariableOp_1�
&batch_normalization_70/batchnorm/mul_2Mul9batch_normalization_70/batchnorm/ReadVariableOp_1:value:0(batch_normalization_70/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_70/batchnorm/mul_2�
1batch_normalization_70/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_70_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_70/batchnorm/ReadVariableOp_2�
$batch_normalization_70/batchnorm/subSub9batch_normalization_70/batchnorm/ReadVariableOp_2:value:0*batch_normalization_70/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_70/batchnorm/sub�
&batch_normalization_70/batchnorm/add_1AddV2*batch_normalization_70/batchnorm/mul_1:z:0(batch_normalization_70/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_70/batchnorm/add_1v
Relu_3Relu*batch_normalization_70/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_3�
dense_59/MatMul/ReadVariableOpReadVariableOp'dense_59_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_59/MatMul/ReadVariableOp�
dense_59/MatMulMatMulRelu_3:activations:0&dense_59/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_59/MatMul�
dense_59/BiasAdd/ReadVariableOpReadVariableOp(dense_59_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_59/BiasAdd/ReadVariableOp�
dense_59/BiasAddBiasAdddense_59/MatMul:product:0'dense_59/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_59/BiasAddt
IdentityIdentitydense_59/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�

NoOpNoOp0^batch_normalization_66/batchnorm/ReadVariableOp2^batch_normalization_66/batchnorm/ReadVariableOp_12^batch_normalization_66/batchnorm/ReadVariableOp_24^batch_normalization_66/batchnorm/mul/ReadVariableOp0^batch_normalization_67/batchnorm/ReadVariableOp2^batch_normalization_67/batchnorm/ReadVariableOp_12^batch_normalization_67/batchnorm/ReadVariableOp_24^batch_normalization_67/batchnorm/mul/ReadVariableOp0^batch_normalization_68/batchnorm/ReadVariableOp2^batch_normalization_68/batchnorm/ReadVariableOp_12^batch_normalization_68/batchnorm/ReadVariableOp_24^batch_normalization_68/batchnorm/mul/ReadVariableOp0^batch_normalization_69/batchnorm/ReadVariableOp2^batch_normalization_69/batchnorm/ReadVariableOp_12^batch_normalization_69/batchnorm/ReadVariableOp_24^batch_normalization_69/batchnorm/mul/ReadVariableOp0^batch_normalization_70/batchnorm/ReadVariableOp2^batch_normalization_70/batchnorm/ReadVariableOp_12^batch_normalization_70/batchnorm/ReadVariableOp_24^batch_normalization_70/batchnorm/mul/ReadVariableOp^dense_55/MatMul/ReadVariableOp^dense_56/MatMul/ReadVariableOp^dense_57/MatMul/ReadVariableOp^dense_58/MatMul/ReadVariableOp ^dense_59/BiasAdd/ReadVariableOp^dense_59/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������: : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_66/batchnorm/ReadVariableOp/batch_normalization_66/batchnorm/ReadVariableOp2f
1batch_normalization_66/batchnorm/ReadVariableOp_11batch_normalization_66/batchnorm/ReadVariableOp_12f
1batch_normalization_66/batchnorm/ReadVariableOp_21batch_normalization_66/batchnorm/ReadVariableOp_22j
3batch_normalization_66/batchnorm/mul/ReadVariableOp3batch_normalization_66/batchnorm/mul/ReadVariableOp2b
/batch_normalization_67/batchnorm/ReadVariableOp/batch_normalization_67/batchnorm/ReadVariableOp2f
1batch_normalization_67/batchnorm/ReadVariableOp_11batch_normalization_67/batchnorm/ReadVariableOp_12f
1batch_normalization_67/batchnorm/ReadVariableOp_21batch_normalization_67/batchnorm/ReadVariableOp_22j
3batch_normalization_67/batchnorm/mul/ReadVariableOp3batch_normalization_67/batchnorm/mul/ReadVariableOp2b
/batch_normalization_68/batchnorm/ReadVariableOp/batch_normalization_68/batchnorm/ReadVariableOp2f
1batch_normalization_68/batchnorm/ReadVariableOp_11batch_normalization_68/batchnorm/ReadVariableOp_12f
1batch_normalization_68/batchnorm/ReadVariableOp_21batch_normalization_68/batchnorm/ReadVariableOp_22j
3batch_normalization_68/batchnorm/mul/ReadVariableOp3batch_normalization_68/batchnorm/mul/ReadVariableOp2b
/batch_normalization_69/batchnorm/ReadVariableOp/batch_normalization_69/batchnorm/ReadVariableOp2f
1batch_normalization_69/batchnorm/ReadVariableOp_11batch_normalization_69/batchnorm/ReadVariableOp_12f
1batch_normalization_69/batchnorm/ReadVariableOp_21batch_normalization_69/batchnorm/ReadVariableOp_22j
3batch_normalization_69/batchnorm/mul/ReadVariableOp3batch_normalization_69/batchnorm/mul/ReadVariableOp2b
/batch_normalization_70/batchnorm/ReadVariableOp/batch_normalization_70/batchnorm/ReadVariableOp2f
1batch_normalization_70/batchnorm/ReadVariableOp_11batch_normalization_70/batchnorm/ReadVariableOp_12f
1batch_normalization_70/batchnorm/ReadVariableOp_21batch_normalization_70/batchnorm/ReadVariableOp_22j
3batch_normalization_70/batchnorm/mul/ReadVariableOp3batch_normalization_70/batchnorm/mul/ReadVariableOp2@
dense_55/MatMul/ReadVariableOpdense_55/MatMul/ReadVariableOp2@
dense_56/MatMul/ReadVariableOpdense_56/MatMul/ReadVariableOp2@
dense_57/MatMul/ReadVariableOpdense_57/MatMul/ReadVariableOp2@
dense_58/MatMul/ReadVariableOpdense_58/MatMul/ReadVariableOp2B
dense_59/BiasAdd/ReadVariableOpdense_59/BiasAdd/ReadVariableOp2@
dense_59/MatMul/ReadVariableOpdense_59/MatMul/ReadVariableOp:J F
'
_output_shapes
:���������

_user_specified_namex
�
~
*__inference_dense_56_layer_call_fn_7042377

inputs
unknown:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_56_layer_call_and_return_conditional_losses_70405242
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_69_layer_call_and_return_conditional_losses_7042238

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
*__inference_dense_59_layer_call_fn_7042421

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_59_layer_call_and_return_conditional_losses_70405902
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�,
�
S__inference_batch_normalization_66_layer_call_and_return_conditional_losses_7039736

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
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

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������2
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

:*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze�
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
:*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2
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
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�"
"__inference__wrapped_model_7039650
input_1^
Pfeed_forward_sub_net_11_batch_normalization_66_batchnorm_readvariableop_resource:b
Tfeed_forward_sub_net_11_batch_normalization_66_batchnorm_mul_readvariableop_resource:`
Rfeed_forward_sub_net_11_batch_normalization_66_batchnorm_readvariableop_1_resource:`
Rfeed_forward_sub_net_11_batch_normalization_66_batchnorm_readvariableop_2_resource:Q
?feed_forward_sub_net_11_dense_55_matmul_readvariableop_resource:^
Pfeed_forward_sub_net_11_batch_normalization_67_batchnorm_readvariableop_resource:b
Tfeed_forward_sub_net_11_batch_normalization_67_batchnorm_mul_readvariableop_resource:`
Rfeed_forward_sub_net_11_batch_normalization_67_batchnorm_readvariableop_1_resource:`
Rfeed_forward_sub_net_11_batch_normalization_67_batchnorm_readvariableop_2_resource:Q
?feed_forward_sub_net_11_dense_56_matmul_readvariableop_resource:^
Pfeed_forward_sub_net_11_batch_normalization_68_batchnorm_readvariableop_resource:b
Tfeed_forward_sub_net_11_batch_normalization_68_batchnorm_mul_readvariableop_resource:`
Rfeed_forward_sub_net_11_batch_normalization_68_batchnorm_readvariableop_1_resource:`
Rfeed_forward_sub_net_11_batch_normalization_68_batchnorm_readvariableop_2_resource:Q
?feed_forward_sub_net_11_dense_57_matmul_readvariableop_resource:^
Pfeed_forward_sub_net_11_batch_normalization_69_batchnorm_readvariableop_resource:b
Tfeed_forward_sub_net_11_batch_normalization_69_batchnorm_mul_readvariableop_resource:`
Rfeed_forward_sub_net_11_batch_normalization_69_batchnorm_readvariableop_1_resource:`
Rfeed_forward_sub_net_11_batch_normalization_69_batchnorm_readvariableop_2_resource:Q
?feed_forward_sub_net_11_dense_58_matmul_readvariableop_resource:^
Pfeed_forward_sub_net_11_batch_normalization_70_batchnorm_readvariableop_resource:b
Tfeed_forward_sub_net_11_batch_normalization_70_batchnorm_mul_readvariableop_resource:`
Rfeed_forward_sub_net_11_batch_normalization_70_batchnorm_readvariableop_1_resource:`
Rfeed_forward_sub_net_11_batch_normalization_70_batchnorm_readvariableop_2_resource:Q
?feed_forward_sub_net_11_dense_59_matmul_readvariableop_resource:N
@feed_forward_sub_net_11_dense_59_biasadd_readvariableop_resource:
identity��Gfeed_forward_sub_net_11/batch_normalization_66/batchnorm/ReadVariableOp�Ifeed_forward_sub_net_11/batch_normalization_66/batchnorm/ReadVariableOp_1�Ifeed_forward_sub_net_11/batch_normalization_66/batchnorm/ReadVariableOp_2�Kfeed_forward_sub_net_11/batch_normalization_66/batchnorm/mul/ReadVariableOp�Gfeed_forward_sub_net_11/batch_normalization_67/batchnorm/ReadVariableOp�Ifeed_forward_sub_net_11/batch_normalization_67/batchnorm/ReadVariableOp_1�Ifeed_forward_sub_net_11/batch_normalization_67/batchnorm/ReadVariableOp_2�Kfeed_forward_sub_net_11/batch_normalization_67/batchnorm/mul/ReadVariableOp�Gfeed_forward_sub_net_11/batch_normalization_68/batchnorm/ReadVariableOp�Ifeed_forward_sub_net_11/batch_normalization_68/batchnorm/ReadVariableOp_1�Ifeed_forward_sub_net_11/batch_normalization_68/batchnorm/ReadVariableOp_2�Kfeed_forward_sub_net_11/batch_normalization_68/batchnorm/mul/ReadVariableOp�Gfeed_forward_sub_net_11/batch_normalization_69/batchnorm/ReadVariableOp�Ifeed_forward_sub_net_11/batch_normalization_69/batchnorm/ReadVariableOp_1�Ifeed_forward_sub_net_11/batch_normalization_69/batchnorm/ReadVariableOp_2�Kfeed_forward_sub_net_11/batch_normalization_69/batchnorm/mul/ReadVariableOp�Gfeed_forward_sub_net_11/batch_normalization_70/batchnorm/ReadVariableOp�Ifeed_forward_sub_net_11/batch_normalization_70/batchnorm/ReadVariableOp_1�Ifeed_forward_sub_net_11/batch_normalization_70/batchnorm/ReadVariableOp_2�Kfeed_forward_sub_net_11/batch_normalization_70/batchnorm/mul/ReadVariableOp�6feed_forward_sub_net_11/dense_55/MatMul/ReadVariableOp�6feed_forward_sub_net_11/dense_56/MatMul/ReadVariableOp�6feed_forward_sub_net_11/dense_57/MatMul/ReadVariableOp�6feed_forward_sub_net_11/dense_58/MatMul/ReadVariableOp�7feed_forward_sub_net_11/dense_59/BiasAdd/ReadVariableOp�6feed_forward_sub_net_11/dense_59/MatMul/ReadVariableOp�
Gfeed_forward_sub_net_11/batch_normalization_66/batchnorm/ReadVariableOpReadVariableOpPfeed_forward_sub_net_11_batch_normalization_66_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02I
Gfeed_forward_sub_net_11/batch_normalization_66/batchnorm/ReadVariableOp�
>feed_forward_sub_net_11/batch_normalization_66/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2@
>feed_forward_sub_net_11/batch_normalization_66/batchnorm/add/y�
<feed_forward_sub_net_11/batch_normalization_66/batchnorm/addAddV2Ofeed_forward_sub_net_11/batch_normalization_66/batchnorm/ReadVariableOp:value:0Gfeed_forward_sub_net_11/batch_normalization_66/batchnorm/add/y:output:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_11/batch_normalization_66/batchnorm/add�
>feed_forward_sub_net_11/batch_normalization_66/batchnorm/RsqrtRsqrt@feed_forward_sub_net_11/batch_normalization_66/batchnorm/add:z:0*
T0*
_output_shapes
:2@
>feed_forward_sub_net_11/batch_normalization_66/batchnorm/Rsqrt�
Kfeed_forward_sub_net_11/batch_normalization_66/batchnorm/mul/ReadVariableOpReadVariableOpTfeed_forward_sub_net_11_batch_normalization_66_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02M
Kfeed_forward_sub_net_11/batch_normalization_66/batchnorm/mul/ReadVariableOp�
<feed_forward_sub_net_11/batch_normalization_66/batchnorm/mulMulBfeed_forward_sub_net_11/batch_normalization_66/batchnorm/Rsqrt:y:0Sfeed_forward_sub_net_11/batch_normalization_66/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_11/batch_normalization_66/batchnorm/mul�
>feed_forward_sub_net_11/batch_normalization_66/batchnorm/mul_1Mulinput_1@feed_forward_sub_net_11/batch_normalization_66/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2@
>feed_forward_sub_net_11/batch_normalization_66/batchnorm/mul_1�
Ifeed_forward_sub_net_11/batch_normalization_66/batchnorm/ReadVariableOp_1ReadVariableOpRfeed_forward_sub_net_11_batch_normalization_66_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02K
Ifeed_forward_sub_net_11/batch_normalization_66/batchnorm/ReadVariableOp_1�
>feed_forward_sub_net_11/batch_normalization_66/batchnorm/mul_2MulQfeed_forward_sub_net_11/batch_normalization_66/batchnorm/ReadVariableOp_1:value:0@feed_forward_sub_net_11/batch_normalization_66/batchnorm/mul:z:0*
T0*
_output_shapes
:2@
>feed_forward_sub_net_11/batch_normalization_66/batchnorm/mul_2�
Ifeed_forward_sub_net_11/batch_normalization_66/batchnorm/ReadVariableOp_2ReadVariableOpRfeed_forward_sub_net_11_batch_normalization_66_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02K
Ifeed_forward_sub_net_11/batch_normalization_66/batchnorm/ReadVariableOp_2�
<feed_forward_sub_net_11/batch_normalization_66/batchnorm/subSubQfeed_forward_sub_net_11/batch_normalization_66/batchnorm/ReadVariableOp_2:value:0Bfeed_forward_sub_net_11/batch_normalization_66/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_11/batch_normalization_66/batchnorm/sub�
>feed_forward_sub_net_11/batch_normalization_66/batchnorm/add_1AddV2Bfeed_forward_sub_net_11/batch_normalization_66/batchnorm/mul_1:z:0@feed_forward_sub_net_11/batch_normalization_66/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2@
>feed_forward_sub_net_11/batch_normalization_66/batchnorm/add_1�
6feed_forward_sub_net_11/dense_55/MatMul/ReadVariableOpReadVariableOp?feed_forward_sub_net_11_dense_55_matmul_readvariableop_resource*
_output_shapes

:*
dtype028
6feed_forward_sub_net_11/dense_55/MatMul/ReadVariableOp�
'feed_forward_sub_net_11/dense_55/MatMulMatMulBfeed_forward_sub_net_11/batch_normalization_66/batchnorm/add_1:z:0>feed_forward_sub_net_11/dense_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2)
'feed_forward_sub_net_11/dense_55/MatMul�
Gfeed_forward_sub_net_11/batch_normalization_67/batchnorm/ReadVariableOpReadVariableOpPfeed_forward_sub_net_11_batch_normalization_67_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02I
Gfeed_forward_sub_net_11/batch_normalization_67/batchnorm/ReadVariableOp�
>feed_forward_sub_net_11/batch_normalization_67/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2@
>feed_forward_sub_net_11/batch_normalization_67/batchnorm/add/y�
<feed_forward_sub_net_11/batch_normalization_67/batchnorm/addAddV2Ofeed_forward_sub_net_11/batch_normalization_67/batchnorm/ReadVariableOp:value:0Gfeed_forward_sub_net_11/batch_normalization_67/batchnorm/add/y:output:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_11/batch_normalization_67/batchnorm/add�
>feed_forward_sub_net_11/batch_normalization_67/batchnorm/RsqrtRsqrt@feed_forward_sub_net_11/batch_normalization_67/batchnorm/add:z:0*
T0*
_output_shapes
:2@
>feed_forward_sub_net_11/batch_normalization_67/batchnorm/Rsqrt�
Kfeed_forward_sub_net_11/batch_normalization_67/batchnorm/mul/ReadVariableOpReadVariableOpTfeed_forward_sub_net_11_batch_normalization_67_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02M
Kfeed_forward_sub_net_11/batch_normalization_67/batchnorm/mul/ReadVariableOp�
<feed_forward_sub_net_11/batch_normalization_67/batchnorm/mulMulBfeed_forward_sub_net_11/batch_normalization_67/batchnorm/Rsqrt:y:0Sfeed_forward_sub_net_11/batch_normalization_67/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_11/batch_normalization_67/batchnorm/mul�
>feed_forward_sub_net_11/batch_normalization_67/batchnorm/mul_1Mul1feed_forward_sub_net_11/dense_55/MatMul:product:0@feed_forward_sub_net_11/batch_normalization_67/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2@
>feed_forward_sub_net_11/batch_normalization_67/batchnorm/mul_1�
Ifeed_forward_sub_net_11/batch_normalization_67/batchnorm/ReadVariableOp_1ReadVariableOpRfeed_forward_sub_net_11_batch_normalization_67_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02K
Ifeed_forward_sub_net_11/batch_normalization_67/batchnorm/ReadVariableOp_1�
>feed_forward_sub_net_11/batch_normalization_67/batchnorm/mul_2MulQfeed_forward_sub_net_11/batch_normalization_67/batchnorm/ReadVariableOp_1:value:0@feed_forward_sub_net_11/batch_normalization_67/batchnorm/mul:z:0*
T0*
_output_shapes
:2@
>feed_forward_sub_net_11/batch_normalization_67/batchnorm/mul_2�
Ifeed_forward_sub_net_11/batch_normalization_67/batchnorm/ReadVariableOp_2ReadVariableOpRfeed_forward_sub_net_11_batch_normalization_67_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02K
Ifeed_forward_sub_net_11/batch_normalization_67/batchnorm/ReadVariableOp_2�
<feed_forward_sub_net_11/batch_normalization_67/batchnorm/subSubQfeed_forward_sub_net_11/batch_normalization_67/batchnorm/ReadVariableOp_2:value:0Bfeed_forward_sub_net_11/batch_normalization_67/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_11/batch_normalization_67/batchnorm/sub�
>feed_forward_sub_net_11/batch_normalization_67/batchnorm/add_1AddV2Bfeed_forward_sub_net_11/batch_normalization_67/batchnorm/mul_1:z:0@feed_forward_sub_net_11/batch_normalization_67/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2@
>feed_forward_sub_net_11/batch_normalization_67/batchnorm/add_1�
feed_forward_sub_net_11/ReluReluBfeed_forward_sub_net_11/batch_normalization_67/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
feed_forward_sub_net_11/Relu�
6feed_forward_sub_net_11/dense_56/MatMul/ReadVariableOpReadVariableOp?feed_forward_sub_net_11_dense_56_matmul_readvariableop_resource*
_output_shapes

:*
dtype028
6feed_forward_sub_net_11/dense_56/MatMul/ReadVariableOp�
'feed_forward_sub_net_11/dense_56/MatMulMatMul*feed_forward_sub_net_11/Relu:activations:0>feed_forward_sub_net_11/dense_56/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2)
'feed_forward_sub_net_11/dense_56/MatMul�
Gfeed_forward_sub_net_11/batch_normalization_68/batchnorm/ReadVariableOpReadVariableOpPfeed_forward_sub_net_11_batch_normalization_68_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02I
Gfeed_forward_sub_net_11/batch_normalization_68/batchnorm/ReadVariableOp�
>feed_forward_sub_net_11/batch_normalization_68/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2@
>feed_forward_sub_net_11/batch_normalization_68/batchnorm/add/y�
<feed_forward_sub_net_11/batch_normalization_68/batchnorm/addAddV2Ofeed_forward_sub_net_11/batch_normalization_68/batchnorm/ReadVariableOp:value:0Gfeed_forward_sub_net_11/batch_normalization_68/batchnorm/add/y:output:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_11/batch_normalization_68/batchnorm/add�
>feed_forward_sub_net_11/batch_normalization_68/batchnorm/RsqrtRsqrt@feed_forward_sub_net_11/batch_normalization_68/batchnorm/add:z:0*
T0*
_output_shapes
:2@
>feed_forward_sub_net_11/batch_normalization_68/batchnorm/Rsqrt�
Kfeed_forward_sub_net_11/batch_normalization_68/batchnorm/mul/ReadVariableOpReadVariableOpTfeed_forward_sub_net_11_batch_normalization_68_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02M
Kfeed_forward_sub_net_11/batch_normalization_68/batchnorm/mul/ReadVariableOp�
<feed_forward_sub_net_11/batch_normalization_68/batchnorm/mulMulBfeed_forward_sub_net_11/batch_normalization_68/batchnorm/Rsqrt:y:0Sfeed_forward_sub_net_11/batch_normalization_68/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_11/batch_normalization_68/batchnorm/mul�
>feed_forward_sub_net_11/batch_normalization_68/batchnorm/mul_1Mul1feed_forward_sub_net_11/dense_56/MatMul:product:0@feed_forward_sub_net_11/batch_normalization_68/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2@
>feed_forward_sub_net_11/batch_normalization_68/batchnorm/mul_1�
Ifeed_forward_sub_net_11/batch_normalization_68/batchnorm/ReadVariableOp_1ReadVariableOpRfeed_forward_sub_net_11_batch_normalization_68_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02K
Ifeed_forward_sub_net_11/batch_normalization_68/batchnorm/ReadVariableOp_1�
>feed_forward_sub_net_11/batch_normalization_68/batchnorm/mul_2MulQfeed_forward_sub_net_11/batch_normalization_68/batchnorm/ReadVariableOp_1:value:0@feed_forward_sub_net_11/batch_normalization_68/batchnorm/mul:z:0*
T0*
_output_shapes
:2@
>feed_forward_sub_net_11/batch_normalization_68/batchnorm/mul_2�
Ifeed_forward_sub_net_11/batch_normalization_68/batchnorm/ReadVariableOp_2ReadVariableOpRfeed_forward_sub_net_11_batch_normalization_68_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02K
Ifeed_forward_sub_net_11/batch_normalization_68/batchnorm/ReadVariableOp_2�
<feed_forward_sub_net_11/batch_normalization_68/batchnorm/subSubQfeed_forward_sub_net_11/batch_normalization_68/batchnorm/ReadVariableOp_2:value:0Bfeed_forward_sub_net_11/batch_normalization_68/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_11/batch_normalization_68/batchnorm/sub�
>feed_forward_sub_net_11/batch_normalization_68/batchnorm/add_1AddV2Bfeed_forward_sub_net_11/batch_normalization_68/batchnorm/mul_1:z:0@feed_forward_sub_net_11/batch_normalization_68/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2@
>feed_forward_sub_net_11/batch_normalization_68/batchnorm/add_1�
feed_forward_sub_net_11/Relu_1ReluBfeed_forward_sub_net_11/batch_normalization_68/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2 
feed_forward_sub_net_11/Relu_1�
6feed_forward_sub_net_11/dense_57/MatMul/ReadVariableOpReadVariableOp?feed_forward_sub_net_11_dense_57_matmul_readvariableop_resource*
_output_shapes

:*
dtype028
6feed_forward_sub_net_11/dense_57/MatMul/ReadVariableOp�
'feed_forward_sub_net_11/dense_57/MatMulMatMul,feed_forward_sub_net_11/Relu_1:activations:0>feed_forward_sub_net_11/dense_57/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2)
'feed_forward_sub_net_11/dense_57/MatMul�
Gfeed_forward_sub_net_11/batch_normalization_69/batchnorm/ReadVariableOpReadVariableOpPfeed_forward_sub_net_11_batch_normalization_69_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02I
Gfeed_forward_sub_net_11/batch_normalization_69/batchnorm/ReadVariableOp�
>feed_forward_sub_net_11/batch_normalization_69/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2@
>feed_forward_sub_net_11/batch_normalization_69/batchnorm/add/y�
<feed_forward_sub_net_11/batch_normalization_69/batchnorm/addAddV2Ofeed_forward_sub_net_11/batch_normalization_69/batchnorm/ReadVariableOp:value:0Gfeed_forward_sub_net_11/batch_normalization_69/batchnorm/add/y:output:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_11/batch_normalization_69/batchnorm/add�
>feed_forward_sub_net_11/batch_normalization_69/batchnorm/RsqrtRsqrt@feed_forward_sub_net_11/batch_normalization_69/batchnorm/add:z:0*
T0*
_output_shapes
:2@
>feed_forward_sub_net_11/batch_normalization_69/batchnorm/Rsqrt�
Kfeed_forward_sub_net_11/batch_normalization_69/batchnorm/mul/ReadVariableOpReadVariableOpTfeed_forward_sub_net_11_batch_normalization_69_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02M
Kfeed_forward_sub_net_11/batch_normalization_69/batchnorm/mul/ReadVariableOp�
<feed_forward_sub_net_11/batch_normalization_69/batchnorm/mulMulBfeed_forward_sub_net_11/batch_normalization_69/batchnorm/Rsqrt:y:0Sfeed_forward_sub_net_11/batch_normalization_69/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_11/batch_normalization_69/batchnorm/mul�
>feed_forward_sub_net_11/batch_normalization_69/batchnorm/mul_1Mul1feed_forward_sub_net_11/dense_57/MatMul:product:0@feed_forward_sub_net_11/batch_normalization_69/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2@
>feed_forward_sub_net_11/batch_normalization_69/batchnorm/mul_1�
Ifeed_forward_sub_net_11/batch_normalization_69/batchnorm/ReadVariableOp_1ReadVariableOpRfeed_forward_sub_net_11_batch_normalization_69_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02K
Ifeed_forward_sub_net_11/batch_normalization_69/batchnorm/ReadVariableOp_1�
>feed_forward_sub_net_11/batch_normalization_69/batchnorm/mul_2MulQfeed_forward_sub_net_11/batch_normalization_69/batchnorm/ReadVariableOp_1:value:0@feed_forward_sub_net_11/batch_normalization_69/batchnorm/mul:z:0*
T0*
_output_shapes
:2@
>feed_forward_sub_net_11/batch_normalization_69/batchnorm/mul_2�
Ifeed_forward_sub_net_11/batch_normalization_69/batchnorm/ReadVariableOp_2ReadVariableOpRfeed_forward_sub_net_11_batch_normalization_69_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02K
Ifeed_forward_sub_net_11/batch_normalization_69/batchnorm/ReadVariableOp_2�
<feed_forward_sub_net_11/batch_normalization_69/batchnorm/subSubQfeed_forward_sub_net_11/batch_normalization_69/batchnorm/ReadVariableOp_2:value:0Bfeed_forward_sub_net_11/batch_normalization_69/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_11/batch_normalization_69/batchnorm/sub�
>feed_forward_sub_net_11/batch_normalization_69/batchnorm/add_1AddV2Bfeed_forward_sub_net_11/batch_normalization_69/batchnorm/mul_1:z:0@feed_forward_sub_net_11/batch_normalization_69/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2@
>feed_forward_sub_net_11/batch_normalization_69/batchnorm/add_1�
feed_forward_sub_net_11/Relu_2ReluBfeed_forward_sub_net_11/batch_normalization_69/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2 
feed_forward_sub_net_11/Relu_2�
6feed_forward_sub_net_11/dense_58/MatMul/ReadVariableOpReadVariableOp?feed_forward_sub_net_11_dense_58_matmul_readvariableop_resource*
_output_shapes

:*
dtype028
6feed_forward_sub_net_11/dense_58/MatMul/ReadVariableOp�
'feed_forward_sub_net_11/dense_58/MatMulMatMul,feed_forward_sub_net_11/Relu_2:activations:0>feed_forward_sub_net_11/dense_58/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2)
'feed_forward_sub_net_11/dense_58/MatMul�
Gfeed_forward_sub_net_11/batch_normalization_70/batchnorm/ReadVariableOpReadVariableOpPfeed_forward_sub_net_11_batch_normalization_70_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02I
Gfeed_forward_sub_net_11/batch_normalization_70/batchnorm/ReadVariableOp�
>feed_forward_sub_net_11/batch_normalization_70/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2@
>feed_forward_sub_net_11/batch_normalization_70/batchnorm/add/y�
<feed_forward_sub_net_11/batch_normalization_70/batchnorm/addAddV2Ofeed_forward_sub_net_11/batch_normalization_70/batchnorm/ReadVariableOp:value:0Gfeed_forward_sub_net_11/batch_normalization_70/batchnorm/add/y:output:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_11/batch_normalization_70/batchnorm/add�
>feed_forward_sub_net_11/batch_normalization_70/batchnorm/RsqrtRsqrt@feed_forward_sub_net_11/batch_normalization_70/batchnorm/add:z:0*
T0*
_output_shapes
:2@
>feed_forward_sub_net_11/batch_normalization_70/batchnorm/Rsqrt�
Kfeed_forward_sub_net_11/batch_normalization_70/batchnorm/mul/ReadVariableOpReadVariableOpTfeed_forward_sub_net_11_batch_normalization_70_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02M
Kfeed_forward_sub_net_11/batch_normalization_70/batchnorm/mul/ReadVariableOp�
<feed_forward_sub_net_11/batch_normalization_70/batchnorm/mulMulBfeed_forward_sub_net_11/batch_normalization_70/batchnorm/Rsqrt:y:0Sfeed_forward_sub_net_11/batch_normalization_70/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_11/batch_normalization_70/batchnorm/mul�
>feed_forward_sub_net_11/batch_normalization_70/batchnorm/mul_1Mul1feed_forward_sub_net_11/dense_58/MatMul:product:0@feed_forward_sub_net_11/batch_normalization_70/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2@
>feed_forward_sub_net_11/batch_normalization_70/batchnorm/mul_1�
Ifeed_forward_sub_net_11/batch_normalization_70/batchnorm/ReadVariableOp_1ReadVariableOpRfeed_forward_sub_net_11_batch_normalization_70_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02K
Ifeed_forward_sub_net_11/batch_normalization_70/batchnorm/ReadVariableOp_1�
>feed_forward_sub_net_11/batch_normalization_70/batchnorm/mul_2MulQfeed_forward_sub_net_11/batch_normalization_70/batchnorm/ReadVariableOp_1:value:0@feed_forward_sub_net_11/batch_normalization_70/batchnorm/mul:z:0*
T0*
_output_shapes
:2@
>feed_forward_sub_net_11/batch_normalization_70/batchnorm/mul_2�
Ifeed_forward_sub_net_11/batch_normalization_70/batchnorm/ReadVariableOp_2ReadVariableOpRfeed_forward_sub_net_11_batch_normalization_70_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02K
Ifeed_forward_sub_net_11/batch_normalization_70/batchnorm/ReadVariableOp_2�
<feed_forward_sub_net_11/batch_normalization_70/batchnorm/subSubQfeed_forward_sub_net_11/batch_normalization_70/batchnorm/ReadVariableOp_2:value:0Bfeed_forward_sub_net_11/batch_normalization_70/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_11/batch_normalization_70/batchnorm/sub�
>feed_forward_sub_net_11/batch_normalization_70/batchnorm/add_1AddV2Bfeed_forward_sub_net_11/batch_normalization_70/batchnorm/mul_1:z:0@feed_forward_sub_net_11/batch_normalization_70/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2@
>feed_forward_sub_net_11/batch_normalization_70/batchnorm/add_1�
feed_forward_sub_net_11/Relu_3ReluBfeed_forward_sub_net_11/batch_normalization_70/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2 
feed_forward_sub_net_11/Relu_3�
6feed_forward_sub_net_11/dense_59/MatMul/ReadVariableOpReadVariableOp?feed_forward_sub_net_11_dense_59_matmul_readvariableop_resource*
_output_shapes

:*
dtype028
6feed_forward_sub_net_11/dense_59/MatMul/ReadVariableOp�
'feed_forward_sub_net_11/dense_59/MatMulMatMul,feed_forward_sub_net_11/Relu_3:activations:0>feed_forward_sub_net_11/dense_59/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2)
'feed_forward_sub_net_11/dense_59/MatMul�
7feed_forward_sub_net_11/dense_59/BiasAdd/ReadVariableOpReadVariableOp@feed_forward_sub_net_11_dense_59_biasadd_readvariableop_resource*
_output_shapes
:*
dtype029
7feed_forward_sub_net_11/dense_59/BiasAdd/ReadVariableOp�
(feed_forward_sub_net_11/dense_59/BiasAddBiasAdd1feed_forward_sub_net_11/dense_59/MatMul:product:0?feed_forward_sub_net_11/dense_59/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2*
(feed_forward_sub_net_11/dense_59/BiasAdd�
IdentityIdentity1feed_forward_sub_net_11/dense_59/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOpH^feed_forward_sub_net_11/batch_normalization_66/batchnorm/ReadVariableOpJ^feed_forward_sub_net_11/batch_normalization_66/batchnorm/ReadVariableOp_1J^feed_forward_sub_net_11/batch_normalization_66/batchnorm/ReadVariableOp_2L^feed_forward_sub_net_11/batch_normalization_66/batchnorm/mul/ReadVariableOpH^feed_forward_sub_net_11/batch_normalization_67/batchnorm/ReadVariableOpJ^feed_forward_sub_net_11/batch_normalization_67/batchnorm/ReadVariableOp_1J^feed_forward_sub_net_11/batch_normalization_67/batchnorm/ReadVariableOp_2L^feed_forward_sub_net_11/batch_normalization_67/batchnorm/mul/ReadVariableOpH^feed_forward_sub_net_11/batch_normalization_68/batchnorm/ReadVariableOpJ^feed_forward_sub_net_11/batch_normalization_68/batchnorm/ReadVariableOp_1J^feed_forward_sub_net_11/batch_normalization_68/batchnorm/ReadVariableOp_2L^feed_forward_sub_net_11/batch_normalization_68/batchnorm/mul/ReadVariableOpH^feed_forward_sub_net_11/batch_normalization_69/batchnorm/ReadVariableOpJ^feed_forward_sub_net_11/batch_normalization_69/batchnorm/ReadVariableOp_1J^feed_forward_sub_net_11/batch_normalization_69/batchnorm/ReadVariableOp_2L^feed_forward_sub_net_11/batch_normalization_69/batchnorm/mul/ReadVariableOpH^feed_forward_sub_net_11/batch_normalization_70/batchnorm/ReadVariableOpJ^feed_forward_sub_net_11/batch_normalization_70/batchnorm/ReadVariableOp_1J^feed_forward_sub_net_11/batch_normalization_70/batchnorm/ReadVariableOp_2L^feed_forward_sub_net_11/batch_normalization_70/batchnorm/mul/ReadVariableOp7^feed_forward_sub_net_11/dense_55/MatMul/ReadVariableOp7^feed_forward_sub_net_11/dense_56/MatMul/ReadVariableOp7^feed_forward_sub_net_11/dense_57/MatMul/ReadVariableOp7^feed_forward_sub_net_11/dense_58/MatMul/ReadVariableOp8^feed_forward_sub_net_11/dense_59/BiasAdd/ReadVariableOp7^feed_forward_sub_net_11/dense_59/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������: : : : : : : : : : : : : : : : : : : : : : : : : : 2�
Gfeed_forward_sub_net_11/batch_normalization_66/batchnorm/ReadVariableOpGfeed_forward_sub_net_11/batch_normalization_66/batchnorm/ReadVariableOp2�
Ifeed_forward_sub_net_11/batch_normalization_66/batchnorm/ReadVariableOp_1Ifeed_forward_sub_net_11/batch_normalization_66/batchnorm/ReadVariableOp_12�
Ifeed_forward_sub_net_11/batch_normalization_66/batchnorm/ReadVariableOp_2Ifeed_forward_sub_net_11/batch_normalization_66/batchnorm/ReadVariableOp_22�
Kfeed_forward_sub_net_11/batch_normalization_66/batchnorm/mul/ReadVariableOpKfeed_forward_sub_net_11/batch_normalization_66/batchnorm/mul/ReadVariableOp2�
Gfeed_forward_sub_net_11/batch_normalization_67/batchnorm/ReadVariableOpGfeed_forward_sub_net_11/batch_normalization_67/batchnorm/ReadVariableOp2�
Ifeed_forward_sub_net_11/batch_normalization_67/batchnorm/ReadVariableOp_1Ifeed_forward_sub_net_11/batch_normalization_67/batchnorm/ReadVariableOp_12�
Ifeed_forward_sub_net_11/batch_normalization_67/batchnorm/ReadVariableOp_2Ifeed_forward_sub_net_11/batch_normalization_67/batchnorm/ReadVariableOp_22�
Kfeed_forward_sub_net_11/batch_normalization_67/batchnorm/mul/ReadVariableOpKfeed_forward_sub_net_11/batch_normalization_67/batchnorm/mul/ReadVariableOp2�
Gfeed_forward_sub_net_11/batch_normalization_68/batchnorm/ReadVariableOpGfeed_forward_sub_net_11/batch_normalization_68/batchnorm/ReadVariableOp2�
Ifeed_forward_sub_net_11/batch_normalization_68/batchnorm/ReadVariableOp_1Ifeed_forward_sub_net_11/batch_normalization_68/batchnorm/ReadVariableOp_12�
Ifeed_forward_sub_net_11/batch_normalization_68/batchnorm/ReadVariableOp_2Ifeed_forward_sub_net_11/batch_normalization_68/batchnorm/ReadVariableOp_22�
Kfeed_forward_sub_net_11/batch_normalization_68/batchnorm/mul/ReadVariableOpKfeed_forward_sub_net_11/batch_normalization_68/batchnorm/mul/ReadVariableOp2�
Gfeed_forward_sub_net_11/batch_normalization_69/batchnorm/ReadVariableOpGfeed_forward_sub_net_11/batch_normalization_69/batchnorm/ReadVariableOp2�
Ifeed_forward_sub_net_11/batch_normalization_69/batchnorm/ReadVariableOp_1Ifeed_forward_sub_net_11/batch_normalization_69/batchnorm/ReadVariableOp_12�
Ifeed_forward_sub_net_11/batch_normalization_69/batchnorm/ReadVariableOp_2Ifeed_forward_sub_net_11/batch_normalization_69/batchnorm/ReadVariableOp_22�
Kfeed_forward_sub_net_11/batch_normalization_69/batchnorm/mul/ReadVariableOpKfeed_forward_sub_net_11/batch_normalization_69/batchnorm/mul/ReadVariableOp2�
Gfeed_forward_sub_net_11/batch_normalization_70/batchnorm/ReadVariableOpGfeed_forward_sub_net_11/batch_normalization_70/batchnorm/ReadVariableOp2�
Ifeed_forward_sub_net_11/batch_normalization_70/batchnorm/ReadVariableOp_1Ifeed_forward_sub_net_11/batch_normalization_70/batchnorm/ReadVariableOp_12�
Ifeed_forward_sub_net_11/batch_normalization_70/batchnorm/ReadVariableOp_2Ifeed_forward_sub_net_11/batch_normalization_70/batchnorm/ReadVariableOp_22�
Kfeed_forward_sub_net_11/batch_normalization_70/batchnorm/mul/ReadVariableOpKfeed_forward_sub_net_11/batch_normalization_70/batchnorm/mul/ReadVariableOp2p
6feed_forward_sub_net_11/dense_55/MatMul/ReadVariableOp6feed_forward_sub_net_11/dense_55/MatMul/ReadVariableOp2p
6feed_forward_sub_net_11/dense_56/MatMul/ReadVariableOp6feed_forward_sub_net_11/dense_56/MatMul/ReadVariableOp2p
6feed_forward_sub_net_11/dense_57/MatMul/ReadVariableOp6feed_forward_sub_net_11/dense_57/MatMul/ReadVariableOp2p
6feed_forward_sub_net_11/dense_58/MatMul/ReadVariableOp6feed_forward_sub_net_11/dense_58/MatMul/ReadVariableOp2r
7feed_forward_sub_net_11/dense_59/BiasAdd/ReadVariableOp7feed_forward_sub_net_11/dense_59/BiasAdd/ReadVariableOp2p
6feed_forward_sub_net_11/dense_59/MatMul/ReadVariableOp6feed_forward_sub_net_11/dense_59/MatMul/ReadVariableOp:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
9__inference_feed_forward_sub_net_11_layer_call_fn_7041305
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
:���������*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_feed_forward_sub_net_11_layer_call_and_return_conditional_losses_70408232
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:���������

_user_specified_namex
�
�
E__inference_dense_57_layer_call_and_return_conditional_losses_7040545

inputs0
matmul_readvariableop_resource:
identity��MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:���������2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�,
�
S__inference_batch_normalization_68_layer_call_and_return_conditional_losses_7042192

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
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

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������2
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

:*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze�
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
:*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2
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
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�,
�
S__inference_batch_normalization_69_layer_call_and_return_conditional_losses_7040234

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
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

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������2
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

:*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze�
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
:*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2
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
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
9__inference_feed_forward_sub_net_11_layer_call_fn_7041362
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
:���������*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_feed_forward_sub_net_11_layer_call_and_return_conditional_losses_70408232
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
8__inference_batch_normalization_69_layer_call_fn_7042205

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_69_layer_call_and_return_conditional_losses_70401722
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_70_layer_call_and_return_conditional_losses_7042320

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_dense_56_layer_call_and_return_conditional_losses_7040524

inputs0
matmul_readvariableop_resource:
identity��MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:���������2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_68_layer_call_fn_7042136

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_68_layer_call_and_return_conditional_losses_70400682
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_67_layer_call_fn_7042041

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_67_layer_call_and_return_conditional_losses_70398402
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
9__inference_feed_forward_sub_net_11_layer_call_fn_7041191
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
:���������*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_feed_forward_sub_net_11_layer_call_and_return_conditional_losses_70405972
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
S__inference_batch_normalization_69_layer_call_and_return_conditional_losses_7040172

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_dense_58_layer_call_and_return_conditional_losses_7040566

inputs0
matmul_readvariableop_resource:
identity��MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:���������2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
~
*__inference_dense_57_layer_call_fn_7042391

inputs
unknown:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_57_layer_call_and_return_conditional_losses_70405452
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�,
�
S__inference_batch_normalization_69_layer_call_and_return_conditional_losses_7042274

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
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

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������2
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

:*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze�
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
:*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2
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
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�E
�
T__inference_feed_forward_sub_net_11_layer_call_and_return_conditional_losses_7040823
x,
batch_normalization_66_7040756:,
batch_normalization_66_7040758:,
batch_normalization_66_7040760:,
batch_normalization_66_7040762:"
dense_55_7040765:,
batch_normalization_67_7040768:,
batch_normalization_67_7040770:,
batch_normalization_67_7040772:,
batch_normalization_67_7040774:"
dense_56_7040778:,
batch_normalization_68_7040781:,
batch_normalization_68_7040783:,
batch_normalization_68_7040785:,
batch_normalization_68_7040787:"
dense_57_7040791:,
batch_normalization_69_7040794:,
batch_normalization_69_7040796:,
batch_normalization_69_7040798:,
batch_normalization_69_7040800:"
dense_58_7040804:,
batch_normalization_70_7040807:,
batch_normalization_70_7040809:,
batch_normalization_70_7040811:,
batch_normalization_70_7040813:"
dense_59_7040817:
dense_59_7040819:
identity��.batch_normalization_66/StatefulPartitionedCall�.batch_normalization_67/StatefulPartitionedCall�.batch_normalization_68/StatefulPartitionedCall�.batch_normalization_69/StatefulPartitionedCall�.batch_normalization_70/StatefulPartitionedCall� dense_55/StatefulPartitionedCall� dense_56/StatefulPartitionedCall� dense_57/StatefulPartitionedCall� dense_58/StatefulPartitionedCall� dense_59/StatefulPartitionedCall�
.batch_normalization_66/StatefulPartitionedCallStatefulPartitionedCallxbatch_normalization_66_7040756batch_normalization_66_7040758batch_normalization_66_7040760batch_normalization_66_7040762*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_66_layer_call_and_return_conditional_losses_703973620
.batch_normalization_66/StatefulPartitionedCall�
 dense_55/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_66/StatefulPartitionedCall:output:0dense_55_7040765*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_55_layer_call_and_return_conditional_losses_70405032"
 dense_55/StatefulPartitionedCall�
.batch_normalization_67/StatefulPartitionedCallStatefulPartitionedCall)dense_55/StatefulPartitionedCall:output:0batch_normalization_67_7040768batch_normalization_67_7040770batch_normalization_67_7040772batch_normalization_67_7040774*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_67_layer_call_and_return_conditional_losses_703990220
.batch_normalization_67/StatefulPartitionedCall
ReluRelu7batch_normalization_67/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
Relu�
 dense_56/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0dense_56_7040778*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_56_layer_call_and_return_conditional_losses_70405242"
 dense_56/StatefulPartitionedCall�
.batch_normalization_68/StatefulPartitionedCallStatefulPartitionedCall)dense_56/StatefulPartitionedCall:output:0batch_normalization_68_7040781batch_normalization_68_7040783batch_normalization_68_7040785batch_normalization_68_7040787*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_68_layer_call_and_return_conditional_losses_704006820
.batch_normalization_68/StatefulPartitionedCall�
Relu_1Relu7batch_normalization_68/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
Relu_1�
 dense_57/StatefulPartitionedCallStatefulPartitionedCallRelu_1:activations:0dense_57_7040791*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_57_layer_call_and_return_conditional_losses_70405452"
 dense_57/StatefulPartitionedCall�
.batch_normalization_69/StatefulPartitionedCallStatefulPartitionedCall)dense_57/StatefulPartitionedCall:output:0batch_normalization_69_7040794batch_normalization_69_7040796batch_normalization_69_7040798batch_normalization_69_7040800*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_69_layer_call_and_return_conditional_losses_704023420
.batch_normalization_69/StatefulPartitionedCall�
Relu_2Relu7batch_normalization_69/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
Relu_2�
 dense_58/StatefulPartitionedCallStatefulPartitionedCallRelu_2:activations:0dense_58_7040804*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_58_layer_call_and_return_conditional_losses_70405662"
 dense_58/StatefulPartitionedCall�
.batch_normalization_70/StatefulPartitionedCallStatefulPartitionedCall)dense_58/StatefulPartitionedCall:output:0batch_normalization_70_7040807batch_normalization_70_7040809batch_normalization_70_7040811batch_normalization_70_7040813*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_70_layer_call_and_return_conditional_losses_704040020
.batch_normalization_70/StatefulPartitionedCall�
Relu_3Relu7batch_normalization_70/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
Relu_3�
 dense_59/StatefulPartitionedCallStatefulPartitionedCallRelu_3:activations:0dense_59_7040817dense_59_7040819*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_59_layer_call_and_return_conditional_losses_70405902"
 dense_59/StatefulPartitionedCall�
IdentityIdentity)dense_59/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp/^batch_normalization_66/StatefulPartitionedCall/^batch_normalization_67/StatefulPartitionedCall/^batch_normalization_68/StatefulPartitionedCall/^batch_normalization_69/StatefulPartitionedCall/^batch_normalization_70/StatefulPartitionedCall!^dense_55/StatefulPartitionedCall!^dense_56/StatefulPartitionedCall!^dense_57/StatefulPartitionedCall!^dense_58/StatefulPartitionedCall!^dense_59/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_66/StatefulPartitionedCall.batch_normalization_66/StatefulPartitionedCall2`
.batch_normalization_67/StatefulPartitionedCall.batch_normalization_67/StatefulPartitionedCall2`
.batch_normalization_68/StatefulPartitionedCall.batch_normalization_68/StatefulPartitionedCall2`
.batch_normalization_69/StatefulPartitionedCall.batch_normalization_69/StatefulPartitionedCall2`
.batch_normalization_70/StatefulPartitionedCall.batch_normalization_70/StatefulPartitionedCall2D
 dense_55/StatefulPartitionedCall dense_55/StatefulPartitionedCall2D
 dense_56/StatefulPartitionedCall dense_56/StatefulPartitionedCall2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall2D
 dense_58/StatefulPartitionedCall dense_58/StatefulPartitionedCall2D
 dense_59/StatefulPartitionedCall dense_59/StatefulPartitionedCall:J F
'
_output_shapes
:���������

_user_specified_namex
�
�
9__inference_feed_forward_sub_net_11_layer_call_fn_7041248
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
:���������*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_feed_forward_sub_net_11_layer_call_and_return_conditional_losses_70405972
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:���������

_user_specified_namex
�
�
%__inference_signature_wrapper_7041134
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
:���������*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__wrapped_model_70396502
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
S__inference_batch_normalization_70_layer_call_and_return_conditional_losses_7040338

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_66_layer_call_and_return_conditional_losses_7039674

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
#__inference__traced_restore_7042620
file_prefixe
Wassignvariableop_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_66_gamma:f
Xassignvariableop_1_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_66_beta:g
Yassignvariableop_2_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_67_gamma:f
Xassignvariableop_3_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_67_beta:g
Yassignvariableop_4_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_68_gamma:f
Xassignvariableop_5_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_68_beta:g
Yassignvariableop_6_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_69_gamma:f
Xassignvariableop_7_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_69_beta:g
Yassignvariableop_8_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_70_gamma:f
Xassignvariableop_9_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_70_beta:n
`assignvariableop_10_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_66_moving_mean:r
dassignvariableop_11_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_66_moving_variance:n
`assignvariableop_12_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_67_moving_mean:r
dassignvariableop_13_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_67_moving_variance:n
`assignvariableop_14_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_68_moving_mean:r
dassignvariableop_15_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_68_moving_variance:n
`assignvariableop_16_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_69_moving_mean:r
dassignvariableop_17_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_69_moving_variance:n
`assignvariableop_18_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_70_moving_mean:r
dassignvariableop_19_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_70_moving_variance:_
Massignvariableop_20_nonshared_model_1_feed_forward_sub_net_11_dense_55_kernel:_
Massignvariableop_21_nonshared_model_1_feed_forward_sub_net_11_dense_56_kernel:_
Massignvariableop_22_nonshared_model_1_feed_forward_sub_net_11_dense_57_kernel:_
Massignvariableop_23_nonshared_model_1_feed_forward_sub_net_11_dense_58_kernel:_
Massignvariableop_24_nonshared_model_1_feed_forward_sub_net_11_dense_59_kernel:Y
Kassignvariableop_25_nonshared_model_1_feed_forward_sub_net_11_dense_59_bias:
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
AssignVariableOpAssignVariableOpWassignvariableop_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_66_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpXassignvariableop_1_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_66_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpYassignvariableop_2_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_67_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpXassignvariableop_3_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_67_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpYassignvariableop_4_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_68_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpXassignvariableop_5_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_68_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpYassignvariableop_6_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_69_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpXassignvariableop_7_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_69_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpYassignvariableop_8_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_70_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpXassignvariableop_9_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_70_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp`assignvariableop_10_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_66_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpdassignvariableop_11_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_66_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp`assignvariableop_12_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_67_moving_meanIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpdassignvariableop_13_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_67_moving_varianceIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp`assignvariableop_14_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_68_moving_meanIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOpdassignvariableop_15_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_68_moving_varianceIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp`assignvariableop_16_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_69_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOpdassignvariableop_17_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_69_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp`assignvariableop_18_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_70_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOpdassignvariableop_19_nonshared_model_1_feed_forward_sub_net_11_batch_normalization_70_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOpMassignvariableop_20_nonshared_model_1_feed_forward_sub_net_11_dense_55_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOpMassignvariableop_21_nonshared_model_1_feed_forward_sub_net_11_dense_56_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOpMassignvariableop_22_nonshared_model_1_feed_forward_sub_net_11_dense_57_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOpMassignvariableop_23_nonshared_model_1_feed_forward_sub_net_11_dense_58_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOpMassignvariableop_24_nonshared_model_1_feed_forward_sub_net_11_dense_59_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOpKassignvariableop_25_nonshared_model_1_feed_forward_sub_net_11_dense_59_biasIdentity_25:output:0"/device:CPU:0*
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
��
�
T__inference_feed_forward_sub_net_11_layer_call_and_return_conditional_losses_7041654
xL
>batch_normalization_66_assignmovingavg_readvariableop_resource:N
@batch_normalization_66_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_66_batchnorm_mul_readvariableop_resource:F
8batch_normalization_66_batchnorm_readvariableop_resource:9
'dense_55_matmul_readvariableop_resource:L
>batch_normalization_67_assignmovingavg_readvariableop_resource:N
@batch_normalization_67_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_67_batchnorm_mul_readvariableop_resource:F
8batch_normalization_67_batchnorm_readvariableop_resource:9
'dense_56_matmul_readvariableop_resource:L
>batch_normalization_68_assignmovingavg_readvariableop_resource:N
@batch_normalization_68_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_68_batchnorm_mul_readvariableop_resource:F
8batch_normalization_68_batchnorm_readvariableop_resource:9
'dense_57_matmul_readvariableop_resource:L
>batch_normalization_69_assignmovingavg_readvariableop_resource:N
@batch_normalization_69_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_69_batchnorm_mul_readvariableop_resource:F
8batch_normalization_69_batchnorm_readvariableop_resource:9
'dense_58_matmul_readvariableop_resource:L
>batch_normalization_70_assignmovingavg_readvariableop_resource:N
@batch_normalization_70_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_70_batchnorm_mul_readvariableop_resource:F
8batch_normalization_70_batchnorm_readvariableop_resource:9
'dense_59_matmul_readvariableop_resource:6
(dense_59_biasadd_readvariableop_resource:
identity��&batch_normalization_66/AssignMovingAvg�5batch_normalization_66/AssignMovingAvg/ReadVariableOp�(batch_normalization_66/AssignMovingAvg_1�7batch_normalization_66/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_66/batchnorm/ReadVariableOp�3batch_normalization_66/batchnorm/mul/ReadVariableOp�&batch_normalization_67/AssignMovingAvg�5batch_normalization_67/AssignMovingAvg/ReadVariableOp�(batch_normalization_67/AssignMovingAvg_1�7batch_normalization_67/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_67/batchnorm/ReadVariableOp�3batch_normalization_67/batchnorm/mul/ReadVariableOp�&batch_normalization_68/AssignMovingAvg�5batch_normalization_68/AssignMovingAvg/ReadVariableOp�(batch_normalization_68/AssignMovingAvg_1�7batch_normalization_68/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_68/batchnorm/ReadVariableOp�3batch_normalization_68/batchnorm/mul/ReadVariableOp�&batch_normalization_69/AssignMovingAvg�5batch_normalization_69/AssignMovingAvg/ReadVariableOp�(batch_normalization_69/AssignMovingAvg_1�7batch_normalization_69/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_69/batchnorm/ReadVariableOp�3batch_normalization_69/batchnorm/mul/ReadVariableOp�&batch_normalization_70/AssignMovingAvg�5batch_normalization_70/AssignMovingAvg/ReadVariableOp�(batch_normalization_70/AssignMovingAvg_1�7batch_normalization_70/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_70/batchnorm/ReadVariableOp�3batch_normalization_70/batchnorm/mul/ReadVariableOp�dense_55/MatMul/ReadVariableOp�dense_56/MatMul/ReadVariableOp�dense_57/MatMul/ReadVariableOp�dense_58/MatMul/ReadVariableOp�dense_59/BiasAdd/ReadVariableOp�dense_59/MatMul/ReadVariableOp�
5batch_normalization_66/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_66/moments/mean/reduction_indices�
#batch_normalization_66/moments/meanMeanx>batch_normalization_66/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_66/moments/mean�
+batch_normalization_66/moments/StopGradientStopGradient,batch_normalization_66/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_66/moments/StopGradient�
0batch_normalization_66/moments/SquaredDifferenceSquaredDifferencex4batch_normalization_66/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������22
0batch_normalization_66/moments/SquaredDifference�
9batch_normalization_66/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_66/moments/variance/reduction_indices�
'batch_normalization_66/moments/varianceMean4batch_normalization_66/moments/SquaredDifference:z:0Bbatch_normalization_66/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_66/moments/variance�
&batch_normalization_66/moments/SqueezeSqueeze,batch_normalization_66/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_66/moments/Squeeze�
(batch_normalization_66/moments/Squeeze_1Squeeze0batch_normalization_66/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_66/moments/Squeeze_1�
,batch_normalization_66/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_66/AssignMovingAvg/decay�
+batch_normalization_66/AssignMovingAvg/CastCast5batch_normalization_66/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_66/AssignMovingAvg/Cast�
5batch_normalization_66/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_66_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_66/AssignMovingAvg/ReadVariableOp�
*batch_normalization_66/AssignMovingAvg/subSub=batch_normalization_66/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_66/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_66/AssignMovingAvg/sub�
*batch_normalization_66/AssignMovingAvg/mulMul.batch_normalization_66/AssignMovingAvg/sub:z:0/batch_normalization_66/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_66/AssignMovingAvg/mul�
&batch_normalization_66/AssignMovingAvgAssignSubVariableOp>batch_normalization_66_assignmovingavg_readvariableop_resource.batch_normalization_66/AssignMovingAvg/mul:z:06^batch_normalization_66/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_66/AssignMovingAvg�
.batch_normalization_66/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_66/AssignMovingAvg_1/decay�
-batch_normalization_66/AssignMovingAvg_1/CastCast7batch_normalization_66/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_66/AssignMovingAvg_1/Cast�
7batch_normalization_66/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_66_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_66/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_66/AssignMovingAvg_1/subSub?batch_normalization_66/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_66/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_66/AssignMovingAvg_1/sub�
,batch_normalization_66/AssignMovingAvg_1/mulMul0batch_normalization_66/AssignMovingAvg_1/sub:z:01batch_normalization_66/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_66/AssignMovingAvg_1/mul�
(batch_normalization_66/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_66_assignmovingavg_1_readvariableop_resource0batch_normalization_66/AssignMovingAvg_1/mul:z:08^batch_normalization_66/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_66/AssignMovingAvg_1�
&batch_normalization_66/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_66/batchnorm/add/y�
$batch_normalization_66/batchnorm/addAddV21batch_normalization_66/moments/Squeeze_1:output:0/batch_normalization_66/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_66/batchnorm/add�
&batch_normalization_66/batchnorm/RsqrtRsqrt(batch_normalization_66/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_66/batchnorm/Rsqrt�
3batch_normalization_66/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_66_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_66/batchnorm/mul/ReadVariableOp�
$batch_normalization_66/batchnorm/mulMul*batch_normalization_66/batchnorm/Rsqrt:y:0;batch_normalization_66/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_66/batchnorm/mul�
&batch_normalization_66/batchnorm/mul_1Mulx(batch_normalization_66/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_66/batchnorm/mul_1�
&batch_normalization_66/batchnorm/mul_2Mul/batch_normalization_66/moments/Squeeze:output:0(batch_normalization_66/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_66/batchnorm/mul_2�
/batch_normalization_66/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_66_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_66/batchnorm/ReadVariableOp�
$batch_normalization_66/batchnorm/subSub7batch_normalization_66/batchnorm/ReadVariableOp:value:0*batch_normalization_66/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_66/batchnorm/sub�
&batch_normalization_66/batchnorm/add_1AddV2*batch_normalization_66/batchnorm/mul_1:z:0(batch_normalization_66/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_66/batchnorm/add_1�
dense_55/MatMul/ReadVariableOpReadVariableOp'dense_55_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_55/MatMul/ReadVariableOp�
dense_55/MatMulMatMul*batch_normalization_66/batchnorm/add_1:z:0&dense_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_55/MatMul�
5batch_normalization_67/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_67/moments/mean/reduction_indices�
#batch_normalization_67/moments/meanMeandense_55/MatMul:product:0>batch_normalization_67/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_67/moments/mean�
+batch_normalization_67/moments/StopGradientStopGradient,batch_normalization_67/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_67/moments/StopGradient�
0batch_normalization_67/moments/SquaredDifferenceSquaredDifferencedense_55/MatMul:product:04batch_normalization_67/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������22
0batch_normalization_67/moments/SquaredDifference�
9batch_normalization_67/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_67/moments/variance/reduction_indices�
'batch_normalization_67/moments/varianceMean4batch_normalization_67/moments/SquaredDifference:z:0Bbatch_normalization_67/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_67/moments/variance�
&batch_normalization_67/moments/SqueezeSqueeze,batch_normalization_67/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_67/moments/Squeeze�
(batch_normalization_67/moments/Squeeze_1Squeeze0batch_normalization_67/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_67/moments/Squeeze_1�
,batch_normalization_67/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_67/AssignMovingAvg/decay�
+batch_normalization_67/AssignMovingAvg/CastCast5batch_normalization_67/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_67/AssignMovingAvg/Cast�
5batch_normalization_67/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_67_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_67/AssignMovingAvg/ReadVariableOp�
*batch_normalization_67/AssignMovingAvg/subSub=batch_normalization_67/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_67/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_67/AssignMovingAvg/sub�
*batch_normalization_67/AssignMovingAvg/mulMul.batch_normalization_67/AssignMovingAvg/sub:z:0/batch_normalization_67/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_67/AssignMovingAvg/mul�
&batch_normalization_67/AssignMovingAvgAssignSubVariableOp>batch_normalization_67_assignmovingavg_readvariableop_resource.batch_normalization_67/AssignMovingAvg/mul:z:06^batch_normalization_67/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_67/AssignMovingAvg�
.batch_normalization_67/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_67/AssignMovingAvg_1/decay�
-batch_normalization_67/AssignMovingAvg_1/CastCast7batch_normalization_67/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_67/AssignMovingAvg_1/Cast�
7batch_normalization_67/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_67_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_67/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_67/AssignMovingAvg_1/subSub?batch_normalization_67/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_67/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_67/AssignMovingAvg_1/sub�
,batch_normalization_67/AssignMovingAvg_1/mulMul0batch_normalization_67/AssignMovingAvg_1/sub:z:01batch_normalization_67/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_67/AssignMovingAvg_1/mul�
(batch_normalization_67/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_67_assignmovingavg_1_readvariableop_resource0batch_normalization_67/AssignMovingAvg_1/mul:z:08^batch_normalization_67/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_67/AssignMovingAvg_1�
&batch_normalization_67/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_67/batchnorm/add/y�
$batch_normalization_67/batchnorm/addAddV21batch_normalization_67/moments/Squeeze_1:output:0/batch_normalization_67/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_67/batchnorm/add�
&batch_normalization_67/batchnorm/RsqrtRsqrt(batch_normalization_67/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_67/batchnorm/Rsqrt�
3batch_normalization_67/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_67_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_67/batchnorm/mul/ReadVariableOp�
$batch_normalization_67/batchnorm/mulMul*batch_normalization_67/batchnorm/Rsqrt:y:0;batch_normalization_67/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_67/batchnorm/mul�
&batch_normalization_67/batchnorm/mul_1Muldense_55/MatMul:product:0(batch_normalization_67/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_67/batchnorm/mul_1�
&batch_normalization_67/batchnorm/mul_2Mul/batch_normalization_67/moments/Squeeze:output:0(batch_normalization_67/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_67/batchnorm/mul_2�
/batch_normalization_67/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_67_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_67/batchnorm/ReadVariableOp�
$batch_normalization_67/batchnorm/subSub7batch_normalization_67/batchnorm/ReadVariableOp:value:0*batch_normalization_67/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_67/batchnorm/sub�
&batch_normalization_67/batchnorm/add_1AddV2*batch_normalization_67/batchnorm/mul_1:z:0(batch_normalization_67/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_67/batchnorm/add_1r
ReluRelu*batch_normalization_67/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu�
dense_56/MatMul/ReadVariableOpReadVariableOp'dense_56_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_56/MatMul/ReadVariableOp�
dense_56/MatMulMatMulRelu:activations:0&dense_56/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_56/MatMul�
5batch_normalization_68/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_68/moments/mean/reduction_indices�
#batch_normalization_68/moments/meanMeandense_56/MatMul:product:0>batch_normalization_68/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_68/moments/mean�
+batch_normalization_68/moments/StopGradientStopGradient,batch_normalization_68/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_68/moments/StopGradient�
0batch_normalization_68/moments/SquaredDifferenceSquaredDifferencedense_56/MatMul:product:04batch_normalization_68/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������22
0batch_normalization_68/moments/SquaredDifference�
9batch_normalization_68/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_68/moments/variance/reduction_indices�
'batch_normalization_68/moments/varianceMean4batch_normalization_68/moments/SquaredDifference:z:0Bbatch_normalization_68/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_68/moments/variance�
&batch_normalization_68/moments/SqueezeSqueeze,batch_normalization_68/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_68/moments/Squeeze�
(batch_normalization_68/moments/Squeeze_1Squeeze0batch_normalization_68/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_68/moments/Squeeze_1�
,batch_normalization_68/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_68/AssignMovingAvg/decay�
+batch_normalization_68/AssignMovingAvg/CastCast5batch_normalization_68/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_68/AssignMovingAvg/Cast�
5batch_normalization_68/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_68_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_68/AssignMovingAvg/ReadVariableOp�
*batch_normalization_68/AssignMovingAvg/subSub=batch_normalization_68/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_68/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_68/AssignMovingAvg/sub�
*batch_normalization_68/AssignMovingAvg/mulMul.batch_normalization_68/AssignMovingAvg/sub:z:0/batch_normalization_68/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_68/AssignMovingAvg/mul�
&batch_normalization_68/AssignMovingAvgAssignSubVariableOp>batch_normalization_68_assignmovingavg_readvariableop_resource.batch_normalization_68/AssignMovingAvg/mul:z:06^batch_normalization_68/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_68/AssignMovingAvg�
.batch_normalization_68/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_68/AssignMovingAvg_1/decay�
-batch_normalization_68/AssignMovingAvg_1/CastCast7batch_normalization_68/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_68/AssignMovingAvg_1/Cast�
7batch_normalization_68/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_68_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_68/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_68/AssignMovingAvg_1/subSub?batch_normalization_68/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_68/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_68/AssignMovingAvg_1/sub�
,batch_normalization_68/AssignMovingAvg_1/mulMul0batch_normalization_68/AssignMovingAvg_1/sub:z:01batch_normalization_68/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_68/AssignMovingAvg_1/mul�
(batch_normalization_68/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_68_assignmovingavg_1_readvariableop_resource0batch_normalization_68/AssignMovingAvg_1/mul:z:08^batch_normalization_68/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_68/AssignMovingAvg_1�
&batch_normalization_68/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_68/batchnorm/add/y�
$batch_normalization_68/batchnorm/addAddV21batch_normalization_68/moments/Squeeze_1:output:0/batch_normalization_68/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_68/batchnorm/add�
&batch_normalization_68/batchnorm/RsqrtRsqrt(batch_normalization_68/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_68/batchnorm/Rsqrt�
3batch_normalization_68/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_68_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_68/batchnorm/mul/ReadVariableOp�
$batch_normalization_68/batchnorm/mulMul*batch_normalization_68/batchnorm/Rsqrt:y:0;batch_normalization_68/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_68/batchnorm/mul�
&batch_normalization_68/batchnorm/mul_1Muldense_56/MatMul:product:0(batch_normalization_68/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_68/batchnorm/mul_1�
&batch_normalization_68/batchnorm/mul_2Mul/batch_normalization_68/moments/Squeeze:output:0(batch_normalization_68/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_68/batchnorm/mul_2�
/batch_normalization_68/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_68_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_68/batchnorm/ReadVariableOp�
$batch_normalization_68/batchnorm/subSub7batch_normalization_68/batchnorm/ReadVariableOp:value:0*batch_normalization_68/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_68/batchnorm/sub�
&batch_normalization_68/batchnorm/add_1AddV2*batch_normalization_68/batchnorm/mul_1:z:0(batch_normalization_68/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_68/batchnorm/add_1v
Relu_1Relu*batch_normalization_68/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_1�
dense_57/MatMul/ReadVariableOpReadVariableOp'dense_57_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_57/MatMul/ReadVariableOp�
dense_57/MatMulMatMulRelu_1:activations:0&dense_57/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_57/MatMul�
5batch_normalization_69/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_69/moments/mean/reduction_indices�
#batch_normalization_69/moments/meanMeandense_57/MatMul:product:0>batch_normalization_69/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_69/moments/mean�
+batch_normalization_69/moments/StopGradientStopGradient,batch_normalization_69/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_69/moments/StopGradient�
0batch_normalization_69/moments/SquaredDifferenceSquaredDifferencedense_57/MatMul:product:04batch_normalization_69/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������22
0batch_normalization_69/moments/SquaredDifference�
9batch_normalization_69/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_69/moments/variance/reduction_indices�
'batch_normalization_69/moments/varianceMean4batch_normalization_69/moments/SquaredDifference:z:0Bbatch_normalization_69/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_69/moments/variance�
&batch_normalization_69/moments/SqueezeSqueeze,batch_normalization_69/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_69/moments/Squeeze�
(batch_normalization_69/moments/Squeeze_1Squeeze0batch_normalization_69/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_69/moments/Squeeze_1�
,batch_normalization_69/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_69/AssignMovingAvg/decay�
+batch_normalization_69/AssignMovingAvg/CastCast5batch_normalization_69/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_69/AssignMovingAvg/Cast�
5batch_normalization_69/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_69_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_69/AssignMovingAvg/ReadVariableOp�
*batch_normalization_69/AssignMovingAvg/subSub=batch_normalization_69/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_69/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_69/AssignMovingAvg/sub�
*batch_normalization_69/AssignMovingAvg/mulMul.batch_normalization_69/AssignMovingAvg/sub:z:0/batch_normalization_69/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_69/AssignMovingAvg/mul�
&batch_normalization_69/AssignMovingAvgAssignSubVariableOp>batch_normalization_69_assignmovingavg_readvariableop_resource.batch_normalization_69/AssignMovingAvg/mul:z:06^batch_normalization_69/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_69/AssignMovingAvg�
.batch_normalization_69/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_69/AssignMovingAvg_1/decay�
-batch_normalization_69/AssignMovingAvg_1/CastCast7batch_normalization_69/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_69/AssignMovingAvg_1/Cast�
7batch_normalization_69/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_69_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_69/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_69/AssignMovingAvg_1/subSub?batch_normalization_69/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_69/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_69/AssignMovingAvg_1/sub�
,batch_normalization_69/AssignMovingAvg_1/mulMul0batch_normalization_69/AssignMovingAvg_1/sub:z:01batch_normalization_69/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_69/AssignMovingAvg_1/mul�
(batch_normalization_69/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_69_assignmovingavg_1_readvariableop_resource0batch_normalization_69/AssignMovingAvg_1/mul:z:08^batch_normalization_69/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_69/AssignMovingAvg_1�
&batch_normalization_69/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_69/batchnorm/add/y�
$batch_normalization_69/batchnorm/addAddV21batch_normalization_69/moments/Squeeze_1:output:0/batch_normalization_69/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_69/batchnorm/add�
&batch_normalization_69/batchnorm/RsqrtRsqrt(batch_normalization_69/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_69/batchnorm/Rsqrt�
3batch_normalization_69/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_69_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_69/batchnorm/mul/ReadVariableOp�
$batch_normalization_69/batchnorm/mulMul*batch_normalization_69/batchnorm/Rsqrt:y:0;batch_normalization_69/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_69/batchnorm/mul�
&batch_normalization_69/batchnorm/mul_1Muldense_57/MatMul:product:0(batch_normalization_69/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_69/batchnorm/mul_1�
&batch_normalization_69/batchnorm/mul_2Mul/batch_normalization_69/moments/Squeeze:output:0(batch_normalization_69/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_69/batchnorm/mul_2�
/batch_normalization_69/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_69_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_69/batchnorm/ReadVariableOp�
$batch_normalization_69/batchnorm/subSub7batch_normalization_69/batchnorm/ReadVariableOp:value:0*batch_normalization_69/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_69/batchnorm/sub�
&batch_normalization_69/batchnorm/add_1AddV2*batch_normalization_69/batchnorm/mul_1:z:0(batch_normalization_69/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_69/batchnorm/add_1v
Relu_2Relu*batch_normalization_69/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_2�
dense_58/MatMul/ReadVariableOpReadVariableOp'dense_58_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_58/MatMul/ReadVariableOp�
dense_58/MatMulMatMulRelu_2:activations:0&dense_58/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_58/MatMul�
5batch_normalization_70/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_70/moments/mean/reduction_indices�
#batch_normalization_70/moments/meanMeandense_58/MatMul:product:0>batch_normalization_70/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_70/moments/mean�
+batch_normalization_70/moments/StopGradientStopGradient,batch_normalization_70/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_70/moments/StopGradient�
0batch_normalization_70/moments/SquaredDifferenceSquaredDifferencedense_58/MatMul:product:04batch_normalization_70/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������22
0batch_normalization_70/moments/SquaredDifference�
9batch_normalization_70/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_70/moments/variance/reduction_indices�
'batch_normalization_70/moments/varianceMean4batch_normalization_70/moments/SquaredDifference:z:0Bbatch_normalization_70/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_70/moments/variance�
&batch_normalization_70/moments/SqueezeSqueeze,batch_normalization_70/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_70/moments/Squeeze�
(batch_normalization_70/moments/Squeeze_1Squeeze0batch_normalization_70/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_70/moments/Squeeze_1�
,batch_normalization_70/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_70/AssignMovingAvg/decay�
+batch_normalization_70/AssignMovingAvg/CastCast5batch_normalization_70/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_70/AssignMovingAvg/Cast�
5batch_normalization_70/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_70_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_70/AssignMovingAvg/ReadVariableOp�
*batch_normalization_70/AssignMovingAvg/subSub=batch_normalization_70/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_70/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_70/AssignMovingAvg/sub�
*batch_normalization_70/AssignMovingAvg/mulMul.batch_normalization_70/AssignMovingAvg/sub:z:0/batch_normalization_70/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_70/AssignMovingAvg/mul�
&batch_normalization_70/AssignMovingAvgAssignSubVariableOp>batch_normalization_70_assignmovingavg_readvariableop_resource.batch_normalization_70/AssignMovingAvg/mul:z:06^batch_normalization_70/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_70/AssignMovingAvg�
.batch_normalization_70/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_70/AssignMovingAvg_1/decay�
-batch_normalization_70/AssignMovingAvg_1/CastCast7batch_normalization_70/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_70/AssignMovingAvg_1/Cast�
7batch_normalization_70/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_70_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_70/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_70/AssignMovingAvg_1/subSub?batch_normalization_70/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_70/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_70/AssignMovingAvg_1/sub�
,batch_normalization_70/AssignMovingAvg_1/mulMul0batch_normalization_70/AssignMovingAvg_1/sub:z:01batch_normalization_70/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_70/AssignMovingAvg_1/mul�
(batch_normalization_70/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_70_assignmovingavg_1_readvariableop_resource0batch_normalization_70/AssignMovingAvg_1/mul:z:08^batch_normalization_70/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_70/AssignMovingAvg_1�
&batch_normalization_70/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_70/batchnorm/add/y�
$batch_normalization_70/batchnorm/addAddV21batch_normalization_70/moments/Squeeze_1:output:0/batch_normalization_70/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_70/batchnorm/add�
&batch_normalization_70/batchnorm/RsqrtRsqrt(batch_normalization_70/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_70/batchnorm/Rsqrt�
3batch_normalization_70/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_70_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_70/batchnorm/mul/ReadVariableOp�
$batch_normalization_70/batchnorm/mulMul*batch_normalization_70/batchnorm/Rsqrt:y:0;batch_normalization_70/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_70/batchnorm/mul�
&batch_normalization_70/batchnorm/mul_1Muldense_58/MatMul:product:0(batch_normalization_70/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_70/batchnorm/mul_1�
&batch_normalization_70/batchnorm/mul_2Mul/batch_normalization_70/moments/Squeeze:output:0(batch_normalization_70/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_70/batchnorm/mul_2�
/batch_normalization_70/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_70_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_70/batchnorm/ReadVariableOp�
$batch_normalization_70/batchnorm/subSub7batch_normalization_70/batchnorm/ReadVariableOp:value:0*batch_normalization_70/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_70/batchnorm/sub�
&batch_normalization_70/batchnorm/add_1AddV2*batch_normalization_70/batchnorm/mul_1:z:0(batch_normalization_70/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_70/batchnorm/add_1v
Relu_3Relu*batch_normalization_70/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_3�
dense_59/MatMul/ReadVariableOpReadVariableOp'dense_59_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_59/MatMul/ReadVariableOp�
dense_59/MatMulMatMulRelu_3:activations:0&dense_59/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_59/MatMul�
dense_59/BiasAdd/ReadVariableOpReadVariableOp(dense_59_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_59/BiasAdd/ReadVariableOp�
dense_59/BiasAddBiasAdddense_59/MatMul:product:0'dense_59/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_59/BiasAddt
IdentityIdentitydense_59/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp'^batch_normalization_66/AssignMovingAvg6^batch_normalization_66/AssignMovingAvg/ReadVariableOp)^batch_normalization_66/AssignMovingAvg_18^batch_normalization_66/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_66/batchnorm/ReadVariableOp4^batch_normalization_66/batchnorm/mul/ReadVariableOp'^batch_normalization_67/AssignMovingAvg6^batch_normalization_67/AssignMovingAvg/ReadVariableOp)^batch_normalization_67/AssignMovingAvg_18^batch_normalization_67/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_67/batchnorm/ReadVariableOp4^batch_normalization_67/batchnorm/mul/ReadVariableOp'^batch_normalization_68/AssignMovingAvg6^batch_normalization_68/AssignMovingAvg/ReadVariableOp)^batch_normalization_68/AssignMovingAvg_18^batch_normalization_68/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_68/batchnorm/ReadVariableOp4^batch_normalization_68/batchnorm/mul/ReadVariableOp'^batch_normalization_69/AssignMovingAvg6^batch_normalization_69/AssignMovingAvg/ReadVariableOp)^batch_normalization_69/AssignMovingAvg_18^batch_normalization_69/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_69/batchnorm/ReadVariableOp4^batch_normalization_69/batchnorm/mul/ReadVariableOp'^batch_normalization_70/AssignMovingAvg6^batch_normalization_70/AssignMovingAvg/ReadVariableOp)^batch_normalization_70/AssignMovingAvg_18^batch_normalization_70/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_70/batchnorm/ReadVariableOp4^batch_normalization_70/batchnorm/mul/ReadVariableOp^dense_55/MatMul/ReadVariableOp^dense_56/MatMul/ReadVariableOp^dense_57/MatMul/ReadVariableOp^dense_58/MatMul/ReadVariableOp ^dense_59/BiasAdd/ReadVariableOp^dense_59/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������: : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_66/AssignMovingAvg&batch_normalization_66/AssignMovingAvg2n
5batch_normalization_66/AssignMovingAvg/ReadVariableOp5batch_normalization_66/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_66/AssignMovingAvg_1(batch_normalization_66/AssignMovingAvg_12r
7batch_normalization_66/AssignMovingAvg_1/ReadVariableOp7batch_normalization_66/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_66/batchnorm/ReadVariableOp/batch_normalization_66/batchnorm/ReadVariableOp2j
3batch_normalization_66/batchnorm/mul/ReadVariableOp3batch_normalization_66/batchnorm/mul/ReadVariableOp2P
&batch_normalization_67/AssignMovingAvg&batch_normalization_67/AssignMovingAvg2n
5batch_normalization_67/AssignMovingAvg/ReadVariableOp5batch_normalization_67/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_67/AssignMovingAvg_1(batch_normalization_67/AssignMovingAvg_12r
7batch_normalization_67/AssignMovingAvg_1/ReadVariableOp7batch_normalization_67/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_67/batchnorm/ReadVariableOp/batch_normalization_67/batchnorm/ReadVariableOp2j
3batch_normalization_67/batchnorm/mul/ReadVariableOp3batch_normalization_67/batchnorm/mul/ReadVariableOp2P
&batch_normalization_68/AssignMovingAvg&batch_normalization_68/AssignMovingAvg2n
5batch_normalization_68/AssignMovingAvg/ReadVariableOp5batch_normalization_68/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_68/AssignMovingAvg_1(batch_normalization_68/AssignMovingAvg_12r
7batch_normalization_68/AssignMovingAvg_1/ReadVariableOp7batch_normalization_68/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_68/batchnorm/ReadVariableOp/batch_normalization_68/batchnorm/ReadVariableOp2j
3batch_normalization_68/batchnorm/mul/ReadVariableOp3batch_normalization_68/batchnorm/mul/ReadVariableOp2P
&batch_normalization_69/AssignMovingAvg&batch_normalization_69/AssignMovingAvg2n
5batch_normalization_69/AssignMovingAvg/ReadVariableOp5batch_normalization_69/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_69/AssignMovingAvg_1(batch_normalization_69/AssignMovingAvg_12r
7batch_normalization_69/AssignMovingAvg_1/ReadVariableOp7batch_normalization_69/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_69/batchnorm/ReadVariableOp/batch_normalization_69/batchnorm/ReadVariableOp2j
3batch_normalization_69/batchnorm/mul/ReadVariableOp3batch_normalization_69/batchnorm/mul/ReadVariableOp2P
&batch_normalization_70/AssignMovingAvg&batch_normalization_70/AssignMovingAvg2n
5batch_normalization_70/AssignMovingAvg/ReadVariableOp5batch_normalization_70/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_70/AssignMovingAvg_1(batch_normalization_70/AssignMovingAvg_12r
7batch_normalization_70/AssignMovingAvg_1/ReadVariableOp7batch_normalization_70/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_70/batchnorm/ReadVariableOp/batch_normalization_70/batchnorm/ReadVariableOp2j
3batch_normalization_70/batchnorm/mul/ReadVariableOp3batch_normalization_70/batchnorm/mul/ReadVariableOp2@
dense_55/MatMul/ReadVariableOpdense_55/MatMul/ReadVariableOp2@
dense_56/MatMul/ReadVariableOpdense_56/MatMul/ReadVariableOp2@
dense_57/MatMul/ReadVariableOpdense_57/MatMul/ReadVariableOp2@
dense_58/MatMul/ReadVariableOpdense_58/MatMul/ReadVariableOp2B
dense_59/BiasAdd/ReadVariableOpdense_59/BiasAdd/ReadVariableOp2@
dense_59/MatMul/ReadVariableOpdense_59/MatMul/ReadVariableOp:J F
'
_output_shapes
:���������

_user_specified_namex"�L
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
serving_default_input_1:0���������<
output_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
	bn_layers
dense_layers
regularization_losses
	variables
trainable_variables
	keras_api

signatures
�__call__
�_default_save_signature
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
-non_trainable_variables
regularization_losses
.layer_regularization_losses
/metrics
	variables

0layers
trainable_variables
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
T:R2Fnonshared_model_1/feed_forward_sub_net_11/batch_normalization_66/gamma
S:Q2Enonshared_model_1/feed_forward_sub_net_11/batch_normalization_66/beta
T:R2Fnonshared_model_1/feed_forward_sub_net_11/batch_normalization_67/gamma
S:Q2Enonshared_model_1/feed_forward_sub_net_11/batch_normalization_67/beta
T:R2Fnonshared_model_1/feed_forward_sub_net_11/batch_normalization_68/gamma
S:Q2Enonshared_model_1/feed_forward_sub_net_11/batch_normalization_68/beta
T:R2Fnonshared_model_1/feed_forward_sub_net_11/batch_normalization_69/gamma
S:Q2Enonshared_model_1/feed_forward_sub_net_11/batch_normalization_69/beta
T:R2Fnonshared_model_1/feed_forward_sub_net_11/batch_normalization_70/gamma
S:Q2Enonshared_model_1/feed_forward_sub_net_11/batch_normalization_70/beta
\:Z (2Lnonshared_model_1/feed_forward_sub_net_11/batch_normalization_66/moving_mean
`:^ (2Pnonshared_model_1/feed_forward_sub_net_11/batch_normalization_66/moving_variance
\:Z (2Lnonshared_model_1/feed_forward_sub_net_11/batch_normalization_67/moving_mean
`:^ (2Pnonshared_model_1/feed_forward_sub_net_11/batch_normalization_67/moving_variance
\:Z (2Lnonshared_model_1/feed_forward_sub_net_11/batch_normalization_68/moving_mean
`:^ (2Pnonshared_model_1/feed_forward_sub_net_11/batch_normalization_68/moving_variance
\:Z (2Lnonshared_model_1/feed_forward_sub_net_11/batch_normalization_69/moving_mean
`:^ (2Pnonshared_model_1/feed_forward_sub_net_11/batch_normalization_69/moving_variance
\:Z (2Lnonshared_model_1/feed_forward_sub_net_11/batch_normalization_70/moving_mean
`:^ (2Pnonshared_model_1/feed_forward_sub_net_11/batch_normalization_70/moving_variance
K:I29nonshared_model_1/feed_forward_sub_net_11/dense_55/kernel
K:I29nonshared_model_1/feed_forward_sub_net_11/dense_56/kernel
K:I29nonshared_model_1/feed_forward_sub_net_11/dense_57/kernel
K:I29nonshared_model_1/feed_forward_sub_net_11/dense_58/kernel
K:I29nonshared_model_1/feed_forward_sub_net_11/dense_59/kernel
E:C27nonshared_model_1/feed_forward_sub_net_11/dense_59/bias
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
trackable_dict_wrapper
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
`non_trainable_variables
3regularization_losses
alayer_regularization_losses
bmetrics
4	variables

clayers
5trainable_variables
dlayer_metrics
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
enon_trainable_variables
8regularization_losses
flayer_regularization_losses
gmetrics
9	variables

hlayers
:trainable_variables
ilayer_metrics
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
jnon_trainable_variables
=regularization_losses
klayer_regularization_losses
lmetrics
>	variables

mlayers
?trainable_variables
nlayer_metrics
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
onon_trainable_variables
Bregularization_losses
player_regularization_losses
qmetrics
C	variables

rlayers
Dtrainable_variables
slayer_metrics
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
tnon_trainable_variables
Gregularization_losses
ulayer_regularization_losses
vmetrics
H	variables

wlayers
Itrainable_variables
xlayer_metrics
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
ynon_trainable_variables
Lregularization_losses
zlayer_regularization_losses
{metrics
M	variables

|layers
Ntrainable_variables
}layer_metrics
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
~non_trainable_variables
Pregularization_losses
layer_regularization_losses
�metrics
Q	variables
�layers
Rtrainable_variables
�layer_metrics
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
�non_trainable_variables
Tregularization_losses
 �layer_regularization_losses
�metrics
U	variables
�layers
Vtrainable_variables
�layer_metrics
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
�non_trainable_variables
Xregularization_losses
 �layer_regularization_losses
�metrics
Y	variables
�layers
Ztrainable_variables
�layer_metrics
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
�non_trainable_variables
\regularization_losses
 �layer_regularization_losses
�metrics
]	variables
�layers
^trainable_variables
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
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
0
 1"
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
!0
"1"
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
�2�
9__inference_feed_forward_sub_net_11_layer_call_fn_7041191
9__inference_feed_forward_sub_net_11_layer_call_fn_7041248
9__inference_feed_forward_sub_net_11_layer_call_fn_7041305
9__inference_feed_forward_sub_net_11_layer_call_fn_7041362�
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
"__inference__wrapped_model_7039650input_1"�
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
T__inference_feed_forward_sub_net_11_layer_call_and_return_conditional_losses_7041468
T__inference_feed_forward_sub_net_11_layer_call_and_return_conditional_losses_7041654
T__inference_feed_forward_sub_net_11_layer_call_and_return_conditional_losses_7041760
T__inference_feed_forward_sub_net_11_layer_call_and_return_conditional_losses_7041946�
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
%__inference_signature_wrapper_7041134input_1"�
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
8__inference_batch_normalization_66_layer_call_fn_7041959
8__inference_batch_normalization_66_layer_call_fn_7041972�
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
S__inference_batch_normalization_66_layer_call_and_return_conditional_losses_7041992
S__inference_batch_normalization_66_layer_call_and_return_conditional_losses_7042028�
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
8__inference_batch_normalization_67_layer_call_fn_7042041
8__inference_batch_normalization_67_layer_call_fn_7042054�
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
S__inference_batch_normalization_67_layer_call_and_return_conditional_losses_7042074
S__inference_batch_normalization_67_layer_call_and_return_conditional_losses_7042110�
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
8__inference_batch_normalization_68_layer_call_fn_7042123
8__inference_batch_normalization_68_layer_call_fn_7042136�
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
S__inference_batch_normalization_68_layer_call_and_return_conditional_losses_7042156
S__inference_batch_normalization_68_layer_call_and_return_conditional_losses_7042192�
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
8__inference_batch_normalization_69_layer_call_fn_7042205
8__inference_batch_normalization_69_layer_call_fn_7042218�
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
S__inference_batch_normalization_69_layer_call_and_return_conditional_losses_7042238
S__inference_batch_normalization_69_layer_call_and_return_conditional_losses_7042274�
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
8__inference_batch_normalization_70_layer_call_fn_7042287
8__inference_batch_normalization_70_layer_call_fn_7042300�
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
S__inference_batch_normalization_70_layer_call_and_return_conditional_losses_7042320
S__inference_batch_normalization_70_layer_call_and_return_conditional_losses_7042356�
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
*__inference_dense_55_layer_call_fn_7042363�
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
E__inference_dense_55_layer_call_and_return_conditional_losses_7042370�
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
*__inference_dense_56_layer_call_fn_7042377�
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
E__inference_dense_56_layer_call_and_return_conditional_losses_7042384�
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
*__inference_dense_57_layer_call_fn_7042391�
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
E__inference_dense_57_layer_call_and_return_conditional_losses_7042398�
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
*__inference_dense_58_layer_call_fn_7042405�
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
E__inference_dense_58_layer_call_and_return_conditional_losses_7042412�
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
*__inference_dense_59_layer_call_fn_7042421�
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
E__inference_dense_59_layer_call_and_return_conditional_losses_7042431�
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
"__inference__wrapped_model_7039650�' ("!)$#*&%+,0�-
&�#
!�
input_1���������
� "3�0
.
output_1"�
output_1����������
S__inference_batch_normalization_66_layer_call_and_return_conditional_losses_7041992b3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
S__inference_batch_normalization_66_layer_call_and_return_conditional_losses_7042028b3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
8__inference_batch_normalization_66_layer_call_fn_7041959U3�0
)�&
 �
inputs���������
p 
� "�����������
8__inference_batch_normalization_66_layer_call_fn_7041972U3�0
)�&
 �
inputs���������
p
� "�����������
S__inference_batch_normalization_67_layer_call_and_return_conditional_losses_7042074b 3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
S__inference_batch_normalization_67_layer_call_and_return_conditional_losses_7042110b 3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
8__inference_batch_normalization_67_layer_call_fn_7042041U 3�0
)�&
 �
inputs���������
p 
� "�����������
8__inference_batch_normalization_67_layer_call_fn_7042054U 3�0
)�&
 �
inputs���������
p
� "�����������
S__inference_batch_normalization_68_layer_call_and_return_conditional_losses_7042156b"!3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
S__inference_batch_normalization_68_layer_call_and_return_conditional_losses_7042192b!"3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
8__inference_batch_normalization_68_layer_call_fn_7042123U"!3�0
)�&
 �
inputs���������
p 
� "�����������
8__inference_batch_normalization_68_layer_call_fn_7042136U!"3�0
)�&
 �
inputs���������
p
� "�����������
S__inference_batch_normalization_69_layer_call_and_return_conditional_losses_7042238b$#3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
S__inference_batch_normalization_69_layer_call_and_return_conditional_losses_7042274b#$3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
8__inference_batch_normalization_69_layer_call_fn_7042205U$#3�0
)�&
 �
inputs���������
p 
� "�����������
8__inference_batch_normalization_69_layer_call_fn_7042218U#$3�0
)�&
 �
inputs���������
p
� "�����������
S__inference_batch_normalization_70_layer_call_and_return_conditional_losses_7042320b&%3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
S__inference_batch_normalization_70_layer_call_and_return_conditional_losses_7042356b%&3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
8__inference_batch_normalization_70_layer_call_fn_7042287U&%3�0
)�&
 �
inputs���������
p 
� "�����������
8__inference_batch_normalization_70_layer_call_fn_7042300U%&3�0
)�&
 �
inputs���������
p
� "�����������
E__inference_dense_55_layer_call_and_return_conditional_losses_7042370['/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� |
*__inference_dense_55_layer_call_fn_7042363N'/�,
%�"
 �
inputs���������
� "�����������
E__inference_dense_56_layer_call_and_return_conditional_losses_7042384[(/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� |
*__inference_dense_56_layer_call_fn_7042377N(/�,
%�"
 �
inputs���������
� "�����������
E__inference_dense_57_layer_call_and_return_conditional_losses_7042398[)/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� |
*__inference_dense_57_layer_call_fn_7042391N)/�,
%�"
 �
inputs���������
� "�����������
E__inference_dense_58_layer_call_and_return_conditional_losses_7042412[*/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� |
*__inference_dense_58_layer_call_fn_7042405N*/�,
%�"
 �
inputs���������
� "�����������
E__inference_dense_59_layer_call_and_return_conditional_losses_7042431\+,/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� }
*__inference_dense_59_layer_call_fn_7042421O+,/�,
%�"
 �
inputs���������
� "�����������
T__inference_feed_forward_sub_net_11_layer_call_and_return_conditional_losses_7041468s' ("!)$#*&%+,.�+
$�!
�
x���������
p 
� "%�"
�
0���������
� �
T__inference_feed_forward_sub_net_11_layer_call_and_return_conditional_losses_7041654s' (!")#$*%&+,.�+
$�!
�
x���������
p
� "%�"
�
0���������
� �
T__inference_feed_forward_sub_net_11_layer_call_and_return_conditional_losses_7041760y' ("!)$#*&%+,4�1
*�'
!�
input_1���������
p 
� "%�"
�
0���������
� �
T__inference_feed_forward_sub_net_11_layer_call_and_return_conditional_losses_7041946y' (!")#$*%&+,4�1
*�'
!�
input_1���������
p
� "%�"
�
0���������
� �
9__inference_feed_forward_sub_net_11_layer_call_fn_7041191l' ("!)$#*&%+,4�1
*�'
!�
input_1���������
p 
� "�����������
9__inference_feed_forward_sub_net_11_layer_call_fn_7041248f' ("!)$#*&%+,.�+
$�!
�
x���������
p 
� "�����������
9__inference_feed_forward_sub_net_11_layer_call_fn_7041305f' (!")#$*%&+,.�+
$�!
�
x���������
p
� "�����������
9__inference_feed_forward_sub_net_11_layer_call_fn_7041362l' (!")#$*%&+,4�1
*�'
!�
input_1���������
p
� "�����������
%__inference_signature_wrapper_7041134�' ("!)$#*&%+,;�8
� 
1�.
,
input_1!�
input_1���������"3�0
.
output_1"�
output_1���������