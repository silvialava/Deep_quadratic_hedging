??
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
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
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
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
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02unknown8??
?
Fnonshared_model_1/feed_forward_sub_net_14/batch_normalization_84/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*W
shared_nameHFnonshared_model_1/feed_forward_sub_net_14/batch_normalization_84/gamma
?
Znonshared_model_1/feed_forward_sub_net_14/batch_normalization_84/gamma/Read/ReadVariableOpReadVariableOpFnonshared_model_1/feed_forward_sub_net_14/batch_normalization_84/gamma*
_output_shapes
:
*
dtype0
?
Enonshared_model_1/feed_forward_sub_net_14/batch_normalization_84/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*V
shared_nameGEnonshared_model_1/feed_forward_sub_net_14/batch_normalization_84/beta
?
Ynonshared_model_1/feed_forward_sub_net_14/batch_normalization_84/beta/Read/ReadVariableOpReadVariableOpEnonshared_model_1/feed_forward_sub_net_14/batch_normalization_84/beta*
_output_shapes
:
*
dtype0
?
Fnonshared_model_1/feed_forward_sub_net_14/batch_normalization_85/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*W
shared_nameHFnonshared_model_1/feed_forward_sub_net_14/batch_normalization_85/gamma
?
Znonshared_model_1/feed_forward_sub_net_14/batch_normalization_85/gamma/Read/ReadVariableOpReadVariableOpFnonshared_model_1/feed_forward_sub_net_14/batch_normalization_85/gamma*
_output_shapes
:*
dtype0
?
Enonshared_model_1/feed_forward_sub_net_14/batch_normalization_85/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*V
shared_nameGEnonshared_model_1/feed_forward_sub_net_14/batch_normalization_85/beta
?
Ynonshared_model_1/feed_forward_sub_net_14/batch_normalization_85/beta/Read/ReadVariableOpReadVariableOpEnonshared_model_1/feed_forward_sub_net_14/batch_normalization_85/beta*
_output_shapes
:*
dtype0
?
Fnonshared_model_1/feed_forward_sub_net_14/batch_normalization_86/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*W
shared_nameHFnonshared_model_1/feed_forward_sub_net_14/batch_normalization_86/gamma
?
Znonshared_model_1/feed_forward_sub_net_14/batch_normalization_86/gamma/Read/ReadVariableOpReadVariableOpFnonshared_model_1/feed_forward_sub_net_14/batch_normalization_86/gamma*
_output_shapes
:*
dtype0
?
Enonshared_model_1/feed_forward_sub_net_14/batch_normalization_86/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*V
shared_nameGEnonshared_model_1/feed_forward_sub_net_14/batch_normalization_86/beta
?
Ynonshared_model_1/feed_forward_sub_net_14/batch_normalization_86/beta/Read/ReadVariableOpReadVariableOpEnonshared_model_1/feed_forward_sub_net_14/batch_normalization_86/beta*
_output_shapes
:*
dtype0
?
Fnonshared_model_1/feed_forward_sub_net_14/batch_normalization_87/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*W
shared_nameHFnonshared_model_1/feed_forward_sub_net_14/batch_normalization_87/gamma
?
Znonshared_model_1/feed_forward_sub_net_14/batch_normalization_87/gamma/Read/ReadVariableOpReadVariableOpFnonshared_model_1/feed_forward_sub_net_14/batch_normalization_87/gamma*
_output_shapes
:*
dtype0
?
Enonshared_model_1/feed_forward_sub_net_14/batch_normalization_87/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*V
shared_nameGEnonshared_model_1/feed_forward_sub_net_14/batch_normalization_87/beta
?
Ynonshared_model_1/feed_forward_sub_net_14/batch_normalization_87/beta/Read/ReadVariableOpReadVariableOpEnonshared_model_1/feed_forward_sub_net_14/batch_normalization_87/beta*
_output_shapes
:*
dtype0
?
Fnonshared_model_1/feed_forward_sub_net_14/batch_normalization_88/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*W
shared_nameHFnonshared_model_1/feed_forward_sub_net_14/batch_normalization_88/gamma
?
Znonshared_model_1/feed_forward_sub_net_14/batch_normalization_88/gamma/Read/ReadVariableOpReadVariableOpFnonshared_model_1/feed_forward_sub_net_14/batch_normalization_88/gamma*
_output_shapes
:*
dtype0
?
Enonshared_model_1/feed_forward_sub_net_14/batch_normalization_88/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*V
shared_nameGEnonshared_model_1/feed_forward_sub_net_14/batch_normalization_88/beta
?
Ynonshared_model_1/feed_forward_sub_net_14/batch_normalization_88/beta/Read/ReadVariableOpReadVariableOpEnonshared_model_1/feed_forward_sub_net_14/batch_normalization_88/beta*
_output_shapes
:*
dtype0
?
Lnonshared_model_1/feed_forward_sub_net_14/batch_normalization_84/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*]
shared_nameNLnonshared_model_1/feed_forward_sub_net_14/batch_normalization_84/moving_mean
?
`nonshared_model_1/feed_forward_sub_net_14/batch_normalization_84/moving_mean/Read/ReadVariableOpReadVariableOpLnonshared_model_1/feed_forward_sub_net_14/batch_normalization_84/moving_mean*
_output_shapes
:
*
dtype0
?
Pnonshared_model_1/feed_forward_sub_net_14/batch_normalization_84/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*a
shared_nameRPnonshared_model_1/feed_forward_sub_net_14/batch_normalization_84/moving_variance
?
dnonshared_model_1/feed_forward_sub_net_14/batch_normalization_84/moving_variance/Read/ReadVariableOpReadVariableOpPnonshared_model_1/feed_forward_sub_net_14/batch_normalization_84/moving_variance*
_output_shapes
:
*
dtype0
?
Lnonshared_model_1/feed_forward_sub_net_14/batch_normalization_85/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*]
shared_nameNLnonshared_model_1/feed_forward_sub_net_14/batch_normalization_85/moving_mean
?
`nonshared_model_1/feed_forward_sub_net_14/batch_normalization_85/moving_mean/Read/ReadVariableOpReadVariableOpLnonshared_model_1/feed_forward_sub_net_14/batch_normalization_85/moving_mean*
_output_shapes
:*
dtype0
?
Pnonshared_model_1/feed_forward_sub_net_14/batch_normalization_85/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*a
shared_nameRPnonshared_model_1/feed_forward_sub_net_14/batch_normalization_85/moving_variance
?
dnonshared_model_1/feed_forward_sub_net_14/batch_normalization_85/moving_variance/Read/ReadVariableOpReadVariableOpPnonshared_model_1/feed_forward_sub_net_14/batch_normalization_85/moving_variance*
_output_shapes
:*
dtype0
?
Lnonshared_model_1/feed_forward_sub_net_14/batch_normalization_86/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*]
shared_nameNLnonshared_model_1/feed_forward_sub_net_14/batch_normalization_86/moving_mean
?
`nonshared_model_1/feed_forward_sub_net_14/batch_normalization_86/moving_mean/Read/ReadVariableOpReadVariableOpLnonshared_model_1/feed_forward_sub_net_14/batch_normalization_86/moving_mean*
_output_shapes
:*
dtype0
?
Pnonshared_model_1/feed_forward_sub_net_14/batch_normalization_86/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*a
shared_nameRPnonshared_model_1/feed_forward_sub_net_14/batch_normalization_86/moving_variance
?
dnonshared_model_1/feed_forward_sub_net_14/batch_normalization_86/moving_variance/Read/ReadVariableOpReadVariableOpPnonshared_model_1/feed_forward_sub_net_14/batch_normalization_86/moving_variance*
_output_shapes
:*
dtype0
?
Lnonshared_model_1/feed_forward_sub_net_14/batch_normalization_87/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*]
shared_nameNLnonshared_model_1/feed_forward_sub_net_14/batch_normalization_87/moving_mean
?
`nonshared_model_1/feed_forward_sub_net_14/batch_normalization_87/moving_mean/Read/ReadVariableOpReadVariableOpLnonshared_model_1/feed_forward_sub_net_14/batch_normalization_87/moving_mean*
_output_shapes
:*
dtype0
?
Pnonshared_model_1/feed_forward_sub_net_14/batch_normalization_87/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*a
shared_nameRPnonshared_model_1/feed_forward_sub_net_14/batch_normalization_87/moving_variance
?
dnonshared_model_1/feed_forward_sub_net_14/batch_normalization_87/moving_variance/Read/ReadVariableOpReadVariableOpPnonshared_model_1/feed_forward_sub_net_14/batch_normalization_87/moving_variance*
_output_shapes
:*
dtype0
?
Lnonshared_model_1/feed_forward_sub_net_14/batch_normalization_88/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*]
shared_nameNLnonshared_model_1/feed_forward_sub_net_14/batch_normalization_88/moving_mean
?
`nonshared_model_1/feed_forward_sub_net_14/batch_normalization_88/moving_mean/Read/ReadVariableOpReadVariableOpLnonshared_model_1/feed_forward_sub_net_14/batch_normalization_88/moving_mean*
_output_shapes
:*
dtype0
?
Pnonshared_model_1/feed_forward_sub_net_14/batch_normalization_88/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*a
shared_nameRPnonshared_model_1/feed_forward_sub_net_14/batch_normalization_88/moving_variance
?
dnonshared_model_1/feed_forward_sub_net_14/batch_normalization_88/moving_variance/Read/ReadVariableOpReadVariableOpPnonshared_model_1/feed_forward_sub_net_14/batch_normalization_88/moving_variance*
_output_shapes
:*
dtype0
?
9nonshared_model_1/feed_forward_sub_net_14/dense_70/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*J
shared_name;9nonshared_model_1/feed_forward_sub_net_14/dense_70/kernel
?
Mnonshared_model_1/feed_forward_sub_net_14/dense_70/kernel/Read/ReadVariableOpReadVariableOp9nonshared_model_1/feed_forward_sub_net_14/dense_70/kernel*
_output_shapes

:
*
dtype0
?
9nonshared_model_1/feed_forward_sub_net_14/dense_71/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*J
shared_name;9nonshared_model_1/feed_forward_sub_net_14/dense_71/kernel
?
Mnonshared_model_1/feed_forward_sub_net_14/dense_71/kernel/Read/ReadVariableOpReadVariableOp9nonshared_model_1/feed_forward_sub_net_14/dense_71/kernel*
_output_shapes

:*
dtype0
?
9nonshared_model_1/feed_forward_sub_net_14/dense_72/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*J
shared_name;9nonshared_model_1/feed_forward_sub_net_14/dense_72/kernel
?
Mnonshared_model_1/feed_forward_sub_net_14/dense_72/kernel/Read/ReadVariableOpReadVariableOp9nonshared_model_1/feed_forward_sub_net_14/dense_72/kernel*
_output_shapes

:*
dtype0
?
9nonshared_model_1/feed_forward_sub_net_14/dense_73/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*J
shared_name;9nonshared_model_1/feed_forward_sub_net_14/dense_73/kernel
?
Mnonshared_model_1/feed_forward_sub_net_14/dense_73/kernel/Read/ReadVariableOpReadVariableOp9nonshared_model_1/feed_forward_sub_net_14/dense_73/kernel*
_output_shapes

:*
dtype0
?
9nonshared_model_1/feed_forward_sub_net_14/dense_74/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*J
shared_name;9nonshared_model_1/feed_forward_sub_net_14/dense_74/kernel
?
Mnonshared_model_1/feed_forward_sub_net_14/dense_74/kernel/Read/ReadVariableOpReadVariableOp9nonshared_model_1/feed_forward_sub_net_14/dense_74/kernel*
_output_shapes

:
*
dtype0
?
7nonshared_model_1/feed_forward_sub_net_14/dense_74/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*H
shared_name97nonshared_model_1/feed_forward_sub_net_14/dense_74/bias
?
Knonshared_model_1/feed_forward_sub_net_14/dense_74/bias/Read/ReadVariableOpReadVariableOp7nonshared_model_1/feed_forward_sub_net_14/dense_74/bias*
_output_shapes
:
*
dtype0

NoOpNoOp
?>
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?>
value?>B?> B?>
?
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
?
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
?
-layer_metrics
regularization_losses
.metrics
	variables
/layer_regularization_losses

0layers
1non_trainable_variables
trainable_variables
 
?
2axis
	gamma
beta
moving_mean
moving_variance
3regularization_losses
4	variables
5trainable_variables
6	keras_api
?
7axis
	gamma
beta
moving_mean
 moving_variance
8regularization_losses
9	variables
:trainable_variables
;	keras_api
?
<axis
	gamma
beta
!moving_mean
"moving_variance
=regularization_losses
>	variables
?trainable_variables
@	keras_api
?
Aaxis
	gamma
beta
#moving_mean
$moving_variance
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
?
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
??
VARIABLE_VALUEFnonshared_model_1/feed_forward_sub_net_14/batch_normalization_84/gamma&variables/0/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEEnonshared_model_1/feed_forward_sub_net_14/batch_normalization_84/beta&variables/1/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEFnonshared_model_1/feed_forward_sub_net_14/batch_normalization_85/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEEnonshared_model_1/feed_forward_sub_net_14/batch_normalization_85/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEFnonshared_model_1/feed_forward_sub_net_14/batch_normalization_86/gamma&variables/4/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEEnonshared_model_1/feed_forward_sub_net_14/batch_normalization_86/beta&variables/5/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEFnonshared_model_1/feed_forward_sub_net_14/batch_normalization_87/gamma&variables/6/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEEnonshared_model_1/feed_forward_sub_net_14/batch_normalization_87/beta&variables/7/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEFnonshared_model_1/feed_forward_sub_net_14/batch_normalization_88/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEEnonshared_model_1/feed_forward_sub_net_14/batch_normalization_88/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUELnonshared_model_1/feed_forward_sub_net_14/batch_normalization_84/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEPnonshared_model_1/feed_forward_sub_net_14/batch_normalization_84/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUELnonshared_model_1/feed_forward_sub_net_14/batch_normalization_85/moving_mean'variables/12/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEPnonshared_model_1/feed_forward_sub_net_14/batch_normalization_85/moving_variance'variables/13/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUELnonshared_model_1/feed_forward_sub_net_14/batch_normalization_86/moving_mean'variables/14/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEPnonshared_model_1/feed_forward_sub_net_14/batch_normalization_86/moving_variance'variables/15/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUELnonshared_model_1/feed_forward_sub_net_14/batch_normalization_87/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEPnonshared_model_1/feed_forward_sub_net_14/batch_normalization_87/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUELnonshared_model_1/feed_forward_sub_net_14/batch_normalization_88/moving_mean'variables/18/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEPnonshared_model_1/feed_forward_sub_net_14/batch_normalization_88/moving_variance'variables/19/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9nonshared_model_1/feed_forward_sub_net_14/dense_70/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9nonshared_model_1/feed_forward_sub_net_14/dense_71/kernel'variables/21/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9nonshared_model_1/feed_forward_sub_net_14/dense_72/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9nonshared_model_1/feed_forward_sub_net_14/dense_73/kernel'variables/23/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9nonshared_model_1/feed_forward_sub_net_14/dense_74/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE7nonshared_model_1/feed_forward_sub_net_14/dense_74/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
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
?
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
?
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
?
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
?
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
?
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
?
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
?
~layer_metrics
Pregularization_losses
metrics
Q	variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
Rtrainable_variables
 

)0

)0
?
?layer_metrics
Tregularization_losses
?metrics
U	variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
Vtrainable_variables
 

*0

*0
?
?layer_metrics
Xregularization_losses
?metrics
Y	variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
Ztrainable_variables
 

+0
,1

+0
,1
?
?layer_metrics
\regularization_losses
?metrics
]	variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
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
:?????????
*
dtype0*
shape:?????????

?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1Pnonshared_model_1/feed_forward_sub_net_14/batch_normalization_84/moving_varianceFnonshared_model_1/feed_forward_sub_net_14/batch_normalization_84/gammaLnonshared_model_1/feed_forward_sub_net_14/batch_normalization_84/moving_meanEnonshared_model_1/feed_forward_sub_net_14/batch_normalization_84/beta9nonshared_model_1/feed_forward_sub_net_14/dense_70/kernelPnonshared_model_1/feed_forward_sub_net_14/batch_normalization_85/moving_varianceFnonshared_model_1/feed_forward_sub_net_14/batch_normalization_85/gammaLnonshared_model_1/feed_forward_sub_net_14/batch_normalization_85/moving_meanEnonshared_model_1/feed_forward_sub_net_14/batch_normalization_85/beta9nonshared_model_1/feed_forward_sub_net_14/dense_71/kernelPnonshared_model_1/feed_forward_sub_net_14/batch_normalization_86/moving_varianceFnonshared_model_1/feed_forward_sub_net_14/batch_normalization_86/gammaLnonshared_model_1/feed_forward_sub_net_14/batch_normalization_86/moving_meanEnonshared_model_1/feed_forward_sub_net_14/batch_normalization_86/beta9nonshared_model_1/feed_forward_sub_net_14/dense_72/kernelPnonshared_model_1/feed_forward_sub_net_14/batch_normalization_87/moving_varianceFnonshared_model_1/feed_forward_sub_net_14/batch_normalization_87/gammaLnonshared_model_1/feed_forward_sub_net_14/batch_normalization_87/moving_meanEnonshared_model_1/feed_forward_sub_net_14/batch_normalization_87/beta9nonshared_model_1/feed_forward_sub_net_14/dense_73/kernelPnonshared_model_1/feed_forward_sub_net_14/batch_normalization_88/moving_varianceFnonshared_model_1/feed_forward_sub_net_14/batch_normalization_88/gammaLnonshared_model_1/feed_forward_sub_net_14/batch_normalization_88/moving_meanEnonshared_model_1/feed_forward_sub_net_14/batch_normalization_88/beta9nonshared_model_1/feed_forward_sub_net_14/dense_74/kernel7nonshared_model_1/feed_forward_sub_net_14/dense_74/bias*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_7061938
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameZnonshared_model_1/feed_forward_sub_net_14/batch_normalization_84/gamma/Read/ReadVariableOpYnonshared_model_1/feed_forward_sub_net_14/batch_normalization_84/beta/Read/ReadVariableOpZnonshared_model_1/feed_forward_sub_net_14/batch_normalization_85/gamma/Read/ReadVariableOpYnonshared_model_1/feed_forward_sub_net_14/batch_normalization_85/beta/Read/ReadVariableOpZnonshared_model_1/feed_forward_sub_net_14/batch_normalization_86/gamma/Read/ReadVariableOpYnonshared_model_1/feed_forward_sub_net_14/batch_normalization_86/beta/Read/ReadVariableOpZnonshared_model_1/feed_forward_sub_net_14/batch_normalization_87/gamma/Read/ReadVariableOpYnonshared_model_1/feed_forward_sub_net_14/batch_normalization_87/beta/Read/ReadVariableOpZnonshared_model_1/feed_forward_sub_net_14/batch_normalization_88/gamma/Read/ReadVariableOpYnonshared_model_1/feed_forward_sub_net_14/batch_normalization_88/beta/Read/ReadVariableOp`nonshared_model_1/feed_forward_sub_net_14/batch_normalization_84/moving_mean/Read/ReadVariableOpdnonshared_model_1/feed_forward_sub_net_14/batch_normalization_84/moving_variance/Read/ReadVariableOp`nonshared_model_1/feed_forward_sub_net_14/batch_normalization_85/moving_mean/Read/ReadVariableOpdnonshared_model_1/feed_forward_sub_net_14/batch_normalization_85/moving_variance/Read/ReadVariableOp`nonshared_model_1/feed_forward_sub_net_14/batch_normalization_86/moving_mean/Read/ReadVariableOpdnonshared_model_1/feed_forward_sub_net_14/batch_normalization_86/moving_variance/Read/ReadVariableOp`nonshared_model_1/feed_forward_sub_net_14/batch_normalization_87/moving_mean/Read/ReadVariableOpdnonshared_model_1/feed_forward_sub_net_14/batch_normalization_87/moving_variance/Read/ReadVariableOp`nonshared_model_1/feed_forward_sub_net_14/batch_normalization_88/moving_mean/Read/ReadVariableOpdnonshared_model_1/feed_forward_sub_net_14/batch_normalization_88/moving_variance/Read/ReadVariableOpMnonshared_model_1/feed_forward_sub_net_14/dense_70/kernel/Read/ReadVariableOpMnonshared_model_1/feed_forward_sub_net_14/dense_71/kernel/Read/ReadVariableOpMnonshared_model_1/feed_forward_sub_net_14/dense_72/kernel/Read/ReadVariableOpMnonshared_model_1/feed_forward_sub_net_14/dense_73/kernel/Read/ReadVariableOpMnonshared_model_1/feed_forward_sub_net_14/dense_74/kernel/Read/ReadVariableOpKnonshared_model_1/feed_forward_sub_net_14/dense_74/bias/Read/ReadVariableOpConst*'
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
GPU 2J 8? *)
f$R"
 __inference__traced_save_7063336
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameFnonshared_model_1/feed_forward_sub_net_14/batch_normalization_84/gammaEnonshared_model_1/feed_forward_sub_net_14/batch_normalization_84/betaFnonshared_model_1/feed_forward_sub_net_14/batch_normalization_85/gammaEnonshared_model_1/feed_forward_sub_net_14/batch_normalization_85/betaFnonshared_model_1/feed_forward_sub_net_14/batch_normalization_86/gammaEnonshared_model_1/feed_forward_sub_net_14/batch_normalization_86/betaFnonshared_model_1/feed_forward_sub_net_14/batch_normalization_87/gammaEnonshared_model_1/feed_forward_sub_net_14/batch_normalization_87/betaFnonshared_model_1/feed_forward_sub_net_14/batch_normalization_88/gammaEnonshared_model_1/feed_forward_sub_net_14/batch_normalization_88/betaLnonshared_model_1/feed_forward_sub_net_14/batch_normalization_84/moving_meanPnonshared_model_1/feed_forward_sub_net_14/batch_normalization_84/moving_varianceLnonshared_model_1/feed_forward_sub_net_14/batch_normalization_85/moving_meanPnonshared_model_1/feed_forward_sub_net_14/batch_normalization_85/moving_varianceLnonshared_model_1/feed_forward_sub_net_14/batch_normalization_86/moving_meanPnonshared_model_1/feed_forward_sub_net_14/batch_normalization_86/moving_varianceLnonshared_model_1/feed_forward_sub_net_14/batch_normalization_87/moving_meanPnonshared_model_1/feed_forward_sub_net_14/batch_normalization_87/moving_varianceLnonshared_model_1/feed_forward_sub_net_14/batch_normalization_88/moving_meanPnonshared_model_1/feed_forward_sub_net_14/batch_normalization_88/moving_variance9nonshared_model_1/feed_forward_sub_net_14/dense_70/kernel9nonshared_model_1/feed_forward_sub_net_14/dense_71/kernel9nonshared_model_1/feed_forward_sub_net_14/dense_72/kernel9nonshared_model_1/feed_forward_sub_net_14/dense_73/kernel9nonshared_model_1/feed_forward_sub_net_14/dense_74/kernel7nonshared_model_1/feed_forward_sub_net_14/dense_74/bias*&
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
GPU 2J 8? *,
f'R%
#__inference__traced_restore_7063424??
?
?
8__inference_batch_normalization_84_layer_call_fn_7062776

inputs
unknown:

	unknown_0:

	unknown_1:

	unknown_2:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_84_layer_call_and_return_conditional_losses_70605402
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
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
:?????????
: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
*__inference_dense_74_layer_call_fn_7063225

inputs
unknown:

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_74_layer_call_and_return_conditional_losses_70613942
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
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
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_dense_72_layer_call_and_return_conditional_losses_7061349

inputs0
matmul_readvariableop_resource:
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
9__inference_feed_forward_sub_net_14_layer_call_fn_7061995
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
identity??StatefulPartitionedCall?
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
:?????????
*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_feed_forward_sub_net_14_layer_call_and_return_conditional_losses_70614012
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
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
G:?????????
: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????

!
_user_specified_name	input_1
?
?
E__inference_dense_70_layer_call_and_return_conditional_losses_7061307

inputs0
matmul_readvariableop_resource:

identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????
: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_87_layer_call_fn_7063022

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_87_layer_call_and_return_conditional_losses_70610382
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?E
?
T__inference_feed_forward_sub_net_14_layer_call_and_return_conditional_losses_7061627
x,
batch_normalization_84_7061560:
,
batch_normalization_84_7061562:
,
batch_normalization_84_7061564:
,
batch_normalization_84_7061566:
"
dense_70_7061569:
,
batch_normalization_85_7061572:,
batch_normalization_85_7061574:,
batch_normalization_85_7061576:,
batch_normalization_85_7061578:"
dense_71_7061582:,
batch_normalization_86_7061585:,
batch_normalization_86_7061587:,
batch_normalization_86_7061589:,
batch_normalization_86_7061591:"
dense_72_7061595:,
batch_normalization_87_7061598:,
batch_normalization_87_7061600:,
batch_normalization_87_7061602:,
batch_normalization_87_7061604:"
dense_73_7061608:,
batch_normalization_88_7061611:,
batch_normalization_88_7061613:,
batch_normalization_88_7061615:,
batch_normalization_88_7061617:"
dense_74_7061621:

dense_74_7061623:

identity??.batch_normalization_84/StatefulPartitionedCall?.batch_normalization_85/StatefulPartitionedCall?.batch_normalization_86/StatefulPartitionedCall?.batch_normalization_87/StatefulPartitionedCall?.batch_normalization_88/StatefulPartitionedCall? dense_70/StatefulPartitionedCall? dense_71/StatefulPartitionedCall? dense_72/StatefulPartitionedCall? dense_73/StatefulPartitionedCall? dense_74/StatefulPartitionedCall?
.batch_normalization_84/StatefulPartitionedCallStatefulPartitionedCallxbatch_normalization_84_7061560batch_normalization_84_7061562batch_normalization_84_7061564batch_normalization_84_7061566*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_84_layer_call_and_return_conditional_losses_706054020
.batch_normalization_84/StatefulPartitionedCall?
 dense_70/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_84/StatefulPartitionedCall:output:0dense_70_7061569*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_70_layer_call_and_return_conditional_losses_70613072"
 dense_70/StatefulPartitionedCall?
.batch_normalization_85/StatefulPartitionedCallStatefulPartitionedCall)dense_70/StatefulPartitionedCall:output:0batch_normalization_85_7061572batch_normalization_85_7061574batch_normalization_85_7061576batch_normalization_85_7061578*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_85_layer_call_and_return_conditional_losses_706070620
.batch_normalization_85/StatefulPartitionedCall
ReluRelu7batch_normalization_85/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
Relu?
 dense_71/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0dense_71_7061582*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_71_layer_call_and_return_conditional_losses_70613282"
 dense_71/StatefulPartitionedCall?
.batch_normalization_86/StatefulPartitionedCallStatefulPartitionedCall)dense_71/StatefulPartitionedCall:output:0batch_normalization_86_7061585batch_normalization_86_7061587batch_normalization_86_7061589batch_normalization_86_7061591*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_86_layer_call_and_return_conditional_losses_706087220
.batch_normalization_86/StatefulPartitionedCall?
Relu_1Relu7batch_normalization_86/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
Relu_1?
 dense_72/StatefulPartitionedCallStatefulPartitionedCallRelu_1:activations:0dense_72_7061595*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_72_layer_call_and_return_conditional_losses_70613492"
 dense_72/StatefulPartitionedCall?
.batch_normalization_87/StatefulPartitionedCallStatefulPartitionedCall)dense_72/StatefulPartitionedCall:output:0batch_normalization_87_7061598batch_normalization_87_7061600batch_normalization_87_7061602batch_normalization_87_7061604*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_87_layer_call_and_return_conditional_losses_706103820
.batch_normalization_87/StatefulPartitionedCall?
Relu_2Relu7batch_normalization_87/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
Relu_2?
 dense_73/StatefulPartitionedCallStatefulPartitionedCallRelu_2:activations:0dense_73_7061608*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_73_layer_call_and_return_conditional_losses_70613702"
 dense_73/StatefulPartitionedCall?
.batch_normalization_88/StatefulPartitionedCallStatefulPartitionedCall)dense_73/StatefulPartitionedCall:output:0batch_normalization_88_7061611batch_normalization_88_7061613batch_normalization_88_7061615batch_normalization_88_7061617*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_88_layer_call_and_return_conditional_losses_706120420
.batch_normalization_88/StatefulPartitionedCall?
Relu_3Relu7batch_normalization_88/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
Relu_3?
 dense_74/StatefulPartitionedCallStatefulPartitionedCallRelu_3:activations:0dense_74_7061621dense_74_7061623*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_74_layer_call_and_return_conditional_losses_70613942"
 dense_74/StatefulPartitionedCall?
IdentityIdentity)dense_74/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp/^batch_normalization_84/StatefulPartitionedCall/^batch_normalization_85/StatefulPartitionedCall/^batch_normalization_86/StatefulPartitionedCall/^batch_normalization_87/StatefulPartitionedCall/^batch_normalization_88/StatefulPartitionedCall!^dense_70/StatefulPartitionedCall!^dense_71/StatefulPartitionedCall!^dense_72/StatefulPartitionedCall!^dense_73/StatefulPartitionedCall!^dense_74/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????
: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_84/StatefulPartitionedCall.batch_normalization_84/StatefulPartitionedCall2`
.batch_normalization_85/StatefulPartitionedCall.batch_normalization_85/StatefulPartitionedCall2`
.batch_normalization_86/StatefulPartitionedCall.batch_normalization_86/StatefulPartitionedCall2`
.batch_normalization_87/StatefulPartitionedCall.batch_normalization_87/StatefulPartitionedCall2`
.batch_normalization_88/StatefulPartitionedCall.batch_normalization_88/StatefulPartitionedCall2D
 dense_70/StatefulPartitionedCall dense_70/StatefulPartitionedCall2D
 dense_71/StatefulPartitionedCall dense_71/StatefulPartitionedCall2D
 dense_72/StatefulPartitionedCall dense_72/StatefulPartitionedCall2D
 dense_73/StatefulPartitionedCall dense_73/StatefulPartitionedCall2D
 dense_74/StatefulPartitionedCall dense_74/StatefulPartitionedCall:J F
'
_output_shapes
:?????????


_user_specified_namex
?
?
S__inference_batch_normalization_84_layer_call_and_return_conditional_losses_7060478

inputs/
!batchnorm_readvariableop_resource:
3
%batchnorm_mul_readvariableop_resource:
1
#batchnorm_readvariableop_1_resource:
1
#batchnorm_readvariableop_2_resource:

identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
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
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????
2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:
2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????
2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_85_layer_call_and_return_conditional_losses_7060644

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_88_layer_call_fn_7063104

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_88_layer_call_and_return_conditional_losses_70612042
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_86_layer_call_and_return_conditional_losses_7062960

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?M
?
 __inference__traced_save_7063336
file_prefixe
asavev2_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_84_gamma_read_readvariableopd
`savev2_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_84_beta_read_readvariableope
asavev2_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_85_gamma_read_readvariableopd
`savev2_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_85_beta_read_readvariableope
asavev2_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_86_gamma_read_readvariableopd
`savev2_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_86_beta_read_readvariableope
asavev2_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_87_gamma_read_readvariableopd
`savev2_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_87_beta_read_readvariableope
asavev2_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_88_gamma_read_readvariableopd
`savev2_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_88_beta_read_readvariableopk
gsavev2_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_84_moving_mean_read_readvariableopo
ksavev2_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_84_moving_variance_read_readvariableopk
gsavev2_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_85_moving_mean_read_readvariableopo
ksavev2_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_85_moving_variance_read_readvariableopk
gsavev2_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_86_moving_mean_read_readvariableopo
ksavev2_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_86_moving_variance_read_readvariableopk
gsavev2_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_87_moving_mean_read_readvariableopo
ksavev2_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_87_moving_variance_read_readvariableopk
gsavev2_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_88_moving_mean_read_readvariableopo
ksavev2_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_88_moving_variance_read_readvariableopX
Tsavev2_nonshared_model_1_feed_forward_sub_net_14_dense_70_kernel_read_readvariableopX
Tsavev2_nonshared_model_1_feed_forward_sub_net_14_dense_71_kernel_read_readvariableopX
Tsavev2_nonshared_model_1_feed_forward_sub_net_14_dense_72_kernel_read_readvariableopX
Tsavev2_nonshared_model_1_feed_forward_sub_net_14_dense_73_kernel_read_readvariableopX
Tsavev2_nonshared_model_1_feed_forward_sub_net_14_dense_74_kernel_read_readvariableopV
Rsavev2_nonshared_model_1_feed_forward_sub_net_14_dense_74_bias_read_readvariableop
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
:*
dtype0*?
value?B?B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0asavev2_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_84_gamma_read_readvariableop`savev2_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_84_beta_read_readvariableopasavev2_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_85_gamma_read_readvariableop`savev2_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_85_beta_read_readvariableopasavev2_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_86_gamma_read_readvariableop`savev2_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_86_beta_read_readvariableopasavev2_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_87_gamma_read_readvariableop`savev2_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_87_beta_read_readvariableopasavev2_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_88_gamma_read_readvariableop`savev2_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_88_beta_read_readvariableopgsavev2_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_84_moving_mean_read_readvariableopksavev2_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_84_moving_variance_read_readvariableopgsavev2_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_85_moving_mean_read_readvariableopksavev2_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_85_moving_variance_read_readvariableopgsavev2_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_86_moving_mean_read_readvariableopksavev2_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_86_moving_variance_read_readvariableopgsavev2_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_87_moving_mean_read_readvariableopksavev2_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_87_moving_variance_read_readvariableopgsavev2_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_88_moving_mean_read_readvariableopksavev2_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_88_moving_variance_read_readvariableopTsavev2_nonshared_model_1_feed_forward_sub_net_14_dense_70_kernel_read_readvariableopTsavev2_nonshared_model_1_feed_forward_sub_net_14_dense_71_kernel_read_readvariableopTsavev2_nonshared_model_1_feed_forward_sub_net_14_dense_72_kernel_read_readvariableopTsavev2_nonshared_model_1_feed_forward_sub_net_14_dense_73_kernel_read_readvariableopTsavev2_nonshared_model_1_feed_forward_sub_net_14_dense_74_kernel_read_readvariableopRsavev2_nonshared_model_1_feed_forward_sub_net_14_dense_74_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *)
dtypes
22
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :
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
?
?
8__inference_batch_normalization_88_layer_call_fn_7063091

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_88_layer_call_and_return_conditional_losses_70611422
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_85_layer_call_and_return_conditional_losses_7062878

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_87_layer_call_and_return_conditional_losses_7063042

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_86_layer_call_fn_7062927

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_86_layer_call_and_return_conditional_losses_70608102
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?,
?
S__inference_batch_normalization_85_layer_call_and_return_conditional_losses_7062914

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
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
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
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
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg/Cast?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul?
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
?#<2
AssignMovingAvg_1/decay?
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg_1/Cast?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1k
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_87_layer_call_fn_7063009

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_87_layer_call_and_return_conditional_losses_70609762
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
#__inference__traced_restore_7063424
file_prefixe
Wassignvariableop_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_84_gamma:
f
Xassignvariableop_1_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_84_beta:
g
Yassignvariableop_2_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_85_gamma:f
Xassignvariableop_3_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_85_beta:g
Yassignvariableop_4_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_86_gamma:f
Xassignvariableop_5_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_86_beta:g
Yassignvariableop_6_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_87_gamma:f
Xassignvariableop_7_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_87_beta:g
Yassignvariableop_8_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_88_gamma:f
Xassignvariableop_9_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_88_beta:n
`assignvariableop_10_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_84_moving_mean:
r
dassignvariableop_11_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_84_moving_variance:
n
`assignvariableop_12_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_85_moving_mean:r
dassignvariableop_13_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_85_moving_variance:n
`assignvariableop_14_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_86_moving_mean:r
dassignvariableop_15_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_86_moving_variance:n
`assignvariableop_16_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_87_moving_mean:r
dassignvariableop_17_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_87_moving_variance:n
`assignvariableop_18_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_88_moving_mean:r
dassignvariableop_19_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_88_moving_variance:_
Massignvariableop_20_nonshared_model_1_feed_forward_sub_net_14_dense_70_kernel:
_
Massignvariableop_21_nonshared_model_1_feed_forward_sub_net_14_dense_71_kernel:_
Massignvariableop_22_nonshared_model_1_feed_forward_sub_net_14_dense_72_kernel:_
Massignvariableop_23_nonshared_model_1_feed_forward_sub_net_14_dense_73_kernel:_
Massignvariableop_24_nonshared_model_1_feed_forward_sub_net_14_dense_74_kernel:
Y
Kassignvariableop_25_nonshared_model_1_feed_forward_sub_net_14_dense_74_bias:

identity_27??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapesn
l:::::::::::::::::::::::::::*)
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpWassignvariableop_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_84_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpXassignvariableop_1_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_84_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpYassignvariableop_2_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_85_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpXassignvariableop_3_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_85_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpYassignvariableop_4_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_86_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpXassignvariableop_5_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_86_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpYassignvariableop_6_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_87_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpXassignvariableop_7_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_87_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpYassignvariableop_8_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_88_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpXassignvariableop_9_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_88_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp`assignvariableop_10_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_84_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpdassignvariableop_11_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_84_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp`assignvariableop_12_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_85_moving_meanIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpdassignvariableop_13_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_85_moving_varianceIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp`assignvariableop_14_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_86_moving_meanIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpdassignvariableop_15_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_86_moving_varianceIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp`assignvariableop_16_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_87_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpdassignvariableop_17_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_87_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp`assignvariableop_18_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_88_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpdassignvariableop_19_nonshared_model_1_feed_forward_sub_net_14_batch_normalization_88_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpMassignvariableop_20_nonshared_model_1_feed_forward_sub_net_14_dense_70_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpMassignvariableop_21_nonshared_model_1_feed_forward_sub_net_14_dense_71_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpMassignvariableop_22_nonshared_model_1_feed_forward_sub_net_14_dense_72_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpMassignvariableop_23_nonshared_model_1_feed_forward_sub_net_14_dense_73_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpMassignvariableop_24_nonshared_model_1_feed_forward_sub_net_14_dense_74_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpKassignvariableop_25_nonshared_model_1_feed_forward_sub_net_14_dense_74_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_259
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_26Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_26f
Identity_27IdentityIdentity_26:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_27?
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
?
?
E__inference_dense_71_layer_call_and_return_conditional_losses_7063188

inputs0
matmul_readvariableop_resource:
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?,
?
S__inference_batch_normalization_84_layer_call_and_return_conditional_losses_7062832

inputs5
'assignmovingavg_readvariableop_resource:
7
)assignmovingavg_1_readvariableop_resource:
3
%batchnorm_mul_readvariableop_resource:
/
!batchnorm_readvariableop_resource:

identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
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
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????
2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2
moments/Squeeze?
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
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg/Cast?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:
*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:
2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:
2
AssignMovingAvg/mul?
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
?#<2
AssignMovingAvg_1/decay?
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg_1/Cast?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:
*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:
2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:
2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1k
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2
batchnorm/add/y?
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
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????
2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:
2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????
2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_88_layer_call_and_return_conditional_losses_7063124

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_85_layer_call_fn_7062845

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_85_layer_call_and_return_conditional_losses_70606442
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_86_layer_call_fn_7062940

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_86_layer_call_and_return_conditional_losses_70608722
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
T__inference_feed_forward_sub_net_14_layer_call_and_return_conditional_losses_7062458
xL
>batch_normalization_84_assignmovingavg_readvariableop_resource:
N
@batch_normalization_84_assignmovingavg_1_readvariableop_resource:
J
<batch_normalization_84_batchnorm_mul_readvariableop_resource:
F
8batch_normalization_84_batchnorm_readvariableop_resource:
9
'dense_70_matmul_readvariableop_resource:
L
>batch_normalization_85_assignmovingavg_readvariableop_resource:N
@batch_normalization_85_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_85_batchnorm_mul_readvariableop_resource:F
8batch_normalization_85_batchnorm_readvariableop_resource:9
'dense_71_matmul_readvariableop_resource:L
>batch_normalization_86_assignmovingavg_readvariableop_resource:N
@batch_normalization_86_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_86_batchnorm_mul_readvariableop_resource:F
8batch_normalization_86_batchnorm_readvariableop_resource:9
'dense_72_matmul_readvariableop_resource:L
>batch_normalization_87_assignmovingavg_readvariableop_resource:N
@batch_normalization_87_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_87_batchnorm_mul_readvariableop_resource:F
8batch_normalization_87_batchnorm_readvariableop_resource:9
'dense_73_matmul_readvariableop_resource:L
>batch_normalization_88_assignmovingavg_readvariableop_resource:N
@batch_normalization_88_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_88_batchnorm_mul_readvariableop_resource:F
8batch_normalization_88_batchnorm_readvariableop_resource:9
'dense_74_matmul_readvariableop_resource:
6
(dense_74_biasadd_readvariableop_resource:

identity??&batch_normalization_84/AssignMovingAvg?5batch_normalization_84/AssignMovingAvg/ReadVariableOp?(batch_normalization_84/AssignMovingAvg_1?7batch_normalization_84/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_84/batchnorm/ReadVariableOp?3batch_normalization_84/batchnorm/mul/ReadVariableOp?&batch_normalization_85/AssignMovingAvg?5batch_normalization_85/AssignMovingAvg/ReadVariableOp?(batch_normalization_85/AssignMovingAvg_1?7batch_normalization_85/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_85/batchnorm/ReadVariableOp?3batch_normalization_85/batchnorm/mul/ReadVariableOp?&batch_normalization_86/AssignMovingAvg?5batch_normalization_86/AssignMovingAvg/ReadVariableOp?(batch_normalization_86/AssignMovingAvg_1?7batch_normalization_86/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_86/batchnorm/ReadVariableOp?3batch_normalization_86/batchnorm/mul/ReadVariableOp?&batch_normalization_87/AssignMovingAvg?5batch_normalization_87/AssignMovingAvg/ReadVariableOp?(batch_normalization_87/AssignMovingAvg_1?7batch_normalization_87/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_87/batchnorm/ReadVariableOp?3batch_normalization_87/batchnorm/mul/ReadVariableOp?&batch_normalization_88/AssignMovingAvg?5batch_normalization_88/AssignMovingAvg/ReadVariableOp?(batch_normalization_88/AssignMovingAvg_1?7batch_normalization_88/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_88/batchnorm/ReadVariableOp?3batch_normalization_88/batchnorm/mul/ReadVariableOp?dense_70/MatMul/ReadVariableOp?dense_71/MatMul/ReadVariableOp?dense_72/MatMul/ReadVariableOp?dense_73/MatMul/ReadVariableOp?dense_74/BiasAdd/ReadVariableOp?dense_74/MatMul/ReadVariableOp?
5batch_normalization_84/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_84/moments/mean/reduction_indices?
#batch_normalization_84/moments/meanMeanx>batch_normalization_84/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2%
#batch_normalization_84/moments/mean?
+batch_normalization_84/moments/StopGradientStopGradient,batch_normalization_84/moments/mean:output:0*
T0*
_output_shapes

:
2-
+batch_normalization_84/moments/StopGradient?
0batch_normalization_84/moments/SquaredDifferenceSquaredDifferencex4batch_normalization_84/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????
22
0batch_normalization_84/moments/SquaredDifference?
9batch_normalization_84/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_84/moments/variance/reduction_indices?
'batch_normalization_84/moments/varianceMean4batch_normalization_84/moments/SquaredDifference:z:0Bbatch_normalization_84/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2)
'batch_normalization_84/moments/variance?
&batch_normalization_84/moments/SqueezeSqueeze,batch_normalization_84/moments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2(
&batch_normalization_84/moments/Squeeze?
(batch_normalization_84/moments/Squeeze_1Squeeze0batch_normalization_84/moments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2*
(batch_normalization_84/moments/Squeeze_1?
,batch_normalization_84/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_84/AssignMovingAvg/decay?
+batch_normalization_84/AssignMovingAvg/CastCast5batch_normalization_84/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_84/AssignMovingAvg/Cast?
5batch_normalization_84/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_84_assignmovingavg_readvariableop_resource*
_output_shapes
:
*
dtype027
5batch_normalization_84/AssignMovingAvg/ReadVariableOp?
*batch_normalization_84/AssignMovingAvg/subSub=batch_normalization_84/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_84/moments/Squeeze:output:0*
T0*
_output_shapes
:
2,
*batch_normalization_84/AssignMovingAvg/sub?
*batch_normalization_84/AssignMovingAvg/mulMul.batch_normalization_84/AssignMovingAvg/sub:z:0/batch_normalization_84/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:
2,
*batch_normalization_84/AssignMovingAvg/mul?
&batch_normalization_84/AssignMovingAvgAssignSubVariableOp>batch_normalization_84_assignmovingavg_readvariableop_resource.batch_normalization_84/AssignMovingAvg/mul:z:06^batch_normalization_84/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_84/AssignMovingAvg?
.batch_normalization_84/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_84/AssignMovingAvg_1/decay?
-batch_normalization_84/AssignMovingAvg_1/CastCast7batch_normalization_84/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_84/AssignMovingAvg_1/Cast?
7batch_normalization_84/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_84_assignmovingavg_1_readvariableop_resource*
_output_shapes
:
*
dtype029
7batch_normalization_84/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_84/AssignMovingAvg_1/subSub?batch_normalization_84/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_84/moments/Squeeze_1:output:0*
T0*
_output_shapes
:
2.
,batch_normalization_84/AssignMovingAvg_1/sub?
,batch_normalization_84/AssignMovingAvg_1/mulMul0batch_normalization_84/AssignMovingAvg_1/sub:z:01batch_normalization_84/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:
2.
,batch_normalization_84/AssignMovingAvg_1/mul?
(batch_normalization_84/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_84_assignmovingavg_1_readvariableop_resource0batch_normalization_84/AssignMovingAvg_1/mul:z:08^batch_normalization_84/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_84/AssignMovingAvg_1?
&batch_normalization_84/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2(
&batch_normalization_84/batchnorm/add/y?
$batch_normalization_84/batchnorm/addAddV21batch_normalization_84/moments/Squeeze_1:output:0/batch_normalization_84/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2&
$batch_normalization_84/batchnorm/add?
&batch_normalization_84/batchnorm/RsqrtRsqrt(batch_normalization_84/batchnorm/add:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_84/batchnorm/Rsqrt?
3batch_normalization_84/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_84_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype025
3batch_normalization_84/batchnorm/mul/ReadVariableOp?
$batch_normalization_84/batchnorm/mulMul*batch_normalization_84/batchnorm/Rsqrt:y:0;batch_normalization_84/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2&
$batch_normalization_84/batchnorm/mul?
&batch_normalization_84/batchnorm/mul_1Mulx(batch_normalization_84/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????
2(
&batch_normalization_84/batchnorm/mul_1?
&batch_normalization_84/batchnorm/mul_2Mul/batch_normalization_84/moments/Squeeze:output:0(batch_normalization_84/batchnorm/mul:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_84/batchnorm/mul_2?
/batch_normalization_84/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_84_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype021
/batch_normalization_84/batchnorm/ReadVariableOp?
$batch_normalization_84/batchnorm/subSub7batch_normalization_84/batchnorm/ReadVariableOp:value:0*batch_normalization_84/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2&
$batch_normalization_84/batchnorm/sub?
&batch_normalization_84/batchnorm/add_1AddV2*batch_normalization_84/batchnorm/mul_1:z:0(batch_normalization_84/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????
2(
&batch_normalization_84/batchnorm/add_1?
dense_70/MatMul/ReadVariableOpReadVariableOp'dense_70_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_70/MatMul/ReadVariableOp?
dense_70/MatMulMatMul*batch_normalization_84/batchnorm/add_1:z:0&dense_70/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_70/MatMul?
5batch_normalization_85/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_85/moments/mean/reduction_indices?
#batch_normalization_85/moments/meanMeandense_70/MatMul:product:0>batch_normalization_85/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_85/moments/mean?
+batch_normalization_85/moments/StopGradientStopGradient,batch_normalization_85/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_85/moments/StopGradient?
0batch_normalization_85/moments/SquaredDifferenceSquaredDifferencedense_70/MatMul:product:04batch_normalization_85/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????22
0batch_normalization_85/moments/SquaredDifference?
9batch_normalization_85/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_85/moments/variance/reduction_indices?
'batch_normalization_85/moments/varianceMean4batch_normalization_85/moments/SquaredDifference:z:0Bbatch_normalization_85/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_85/moments/variance?
&batch_normalization_85/moments/SqueezeSqueeze,batch_normalization_85/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_85/moments/Squeeze?
(batch_normalization_85/moments/Squeeze_1Squeeze0batch_normalization_85/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_85/moments/Squeeze_1?
,batch_normalization_85/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_85/AssignMovingAvg/decay?
+batch_normalization_85/AssignMovingAvg/CastCast5batch_normalization_85/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_85/AssignMovingAvg/Cast?
5batch_normalization_85/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_85_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_85/AssignMovingAvg/ReadVariableOp?
*batch_normalization_85/AssignMovingAvg/subSub=batch_normalization_85/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_85/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_85/AssignMovingAvg/sub?
*batch_normalization_85/AssignMovingAvg/mulMul.batch_normalization_85/AssignMovingAvg/sub:z:0/batch_normalization_85/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_85/AssignMovingAvg/mul?
&batch_normalization_85/AssignMovingAvgAssignSubVariableOp>batch_normalization_85_assignmovingavg_readvariableop_resource.batch_normalization_85/AssignMovingAvg/mul:z:06^batch_normalization_85/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_85/AssignMovingAvg?
.batch_normalization_85/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_85/AssignMovingAvg_1/decay?
-batch_normalization_85/AssignMovingAvg_1/CastCast7batch_normalization_85/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_85/AssignMovingAvg_1/Cast?
7batch_normalization_85/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_85_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_85/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_85/AssignMovingAvg_1/subSub?batch_normalization_85/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_85/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_85/AssignMovingAvg_1/sub?
,batch_normalization_85/AssignMovingAvg_1/mulMul0batch_normalization_85/AssignMovingAvg_1/sub:z:01batch_normalization_85/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_85/AssignMovingAvg_1/mul?
(batch_normalization_85/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_85_assignmovingavg_1_readvariableop_resource0batch_normalization_85/AssignMovingAvg_1/mul:z:08^batch_normalization_85/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_85/AssignMovingAvg_1?
&batch_normalization_85/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2(
&batch_normalization_85/batchnorm/add/y?
$batch_normalization_85/batchnorm/addAddV21batch_normalization_85/moments/Squeeze_1:output:0/batch_normalization_85/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_85/batchnorm/add?
&batch_normalization_85/batchnorm/RsqrtRsqrt(batch_normalization_85/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_85/batchnorm/Rsqrt?
3batch_normalization_85/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_85_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_85/batchnorm/mul/ReadVariableOp?
$batch_normalization_85/batchnorm/mulMul*batch_normalization_85/batchnorm/Rsqrt:y:0;batch_normalization_85/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_85/batchnorm/mul?
&batch_normalization_85/batchnorm/mul_1Muldense_70/MatMul:product:0(batch_normalization_85/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_85/batchnorm/mul_1?
&batch_normalization_85/batchnorm/mul_2Mul/batch_normalization_85/moments/Squeeze:output:0(batch_normalization_85/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_85/batchnorm/mul_2?
/batch_normalization_85/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_85_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_85/batchnorm/ReadVariableOp?
$batch_normalization_85/batchnorm/subSub7batch_normalization_85/batchnorm/ReadVariableOp:value:0*batch_normalization_85/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_85/batchnorm/sub?
&batch_normalization_85/batchnorm/add_1AddV2*batch_normalization_85/batchnorm/mul_1:z:0(batch_normalization_85/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_85/batchnorm/add_1r
ReluRelu*batch_normalization_85/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu?
dense_71/MatMul/ReadVariableOpReadVariableOp'dense_71_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_71/MatMul/ReadVariableOp?
dense_71/MatMulMatMulRelu:activations:0&dense_71/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_71/MatMul?
5batch_normalization_86/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_86/moments/mean/reduction_indices?
#batch_normalization_86/moments/meanMeandense_71/MatMul:product:0>batch_normalization_86/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_86/moments/mean?
+batch_normalization_86/moments/StopGradientStopGradient,batch_normalization_86/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_86/moments/StopGradient?
0batch_normalization_86/moments/SquaredDifferenceSquaredDifferencedense_71/MatMul:product:04batch_normalization_86/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????22
0batch_normalization_86/moments/SquaredDifference?
9batch_normalization_86/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_86/moments/variance/reduction_indices?
'batch_normalization_86/moments/varianceMean4batch_normalization_86/moments/SquaredDifference:z:0Bbatch_normalization_86/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_86/moments/variance?
&batch_normalization_86/moments/SqueezeSqueeze,batch_normalization_86/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_86/moments/Squeeze?
(batch_normalization_86/moments/Squeeze_1Squeeze0batch_normalization_86/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_86/moments/Squeeze_1?
,batch_normalization_86/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_86/AssignMovingAvg/decay?
+batch_normalization_86/AssignMovingAvg/CastCast5batch_normalization_86/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_86/AssignMovingAvg/Cast?
5batch_normalization_86/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_86_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_86/AssignMovingAvg/ReadVariableOp?
*batch_normalization_86/AssignMovingAvg/subSub=batch_normalization_86/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_86/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_86/AssignMovingAvg/sub?
*batch_normalization_86/AssignMovingAvg/mulMul.batch_normalization_86/AssignMovingAvg/sub:z:0/batch_normalization_86/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_86/AssignMovingAvg/mul?
&batch_normalization_86/AssignMovingAvgAssignSubVariableOp>batch_normalization_86_assignmovingavg_readvariableop_resource.batch_normalization_86/AssignMovingAvg/mul:z:06^batch_normalization_86/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_86/AssignMovingAvg?
.batch_normalization_86/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_86/AssignMovingAvg_1/decay?
-batch_normalization_86/AssignMovingAvg_1/CastCast7batch_normalization_86/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_86/AssignMovingAvg_1/Cast?
7batch_normalization_86/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_86_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_86/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_86/AssignMovingAvg_1/subSub?batch_normalization_86/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_86/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_86/AssignMovingAvg_1/sub?
,batch_normalization_86/AssignMovingAvg_1/mulMul0batch_normalization_86/AssignMovingAvg_1/sub:z:01batch_normalization_86/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_86/AssignMovingAvg_1/mul?
(batch_normalization_86/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_86_assignmovingavg_1_readvariableop_resource0batch_normalization_86/AssignMovingAvg_1/mul:z:08^batch_normalization_86/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_86/AssignMovingAvg_1?
&batch_normalization_86/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2(
&batch_normalization_86/batchnorm/add/y?
$batch_normalization_86/batchnorm/addAddV21batch_normalization_86/moments/Squeeze_1:output:0/batch_normalization_86/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_86/batchnorm/add?
&batch_normalization_86/batchnorm/RsqrtRsqrt(batch_normalization_86/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_86/batchnorm/Rsqrt?
3batch_normalization_86/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_86_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_86/batchnorm/mul/ReadVariableOp?
$batch_normalization_86/batchnorm/mulMul*batch_normalization_86/batchnorm/Rsqrt:y:0;batch_normalization_86/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_86/batchnorm/mul?
&batch_normalization_86/batchnorm/mul_1Muldense_71/MatMul:product:0(batch_normalization_86/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_86/batchnorm/mul_1?
&batch_normalization_86/batchnorm/mul_2Mul/batch_normalization_86/moments/Squeeze:output:0(batch_normalization_86/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_86/batchnorm/mul_2?
/batch_normalization_86/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_86_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_86/batchnorm/ReadVariableOp?
$batch_normalization_86/batchnorm/subSub7batch_normalization_86/batchnorm/ReadVariableOp:value:0*batch_normalization_86/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_86/batchnorm/sub?
&batch_normalization_86/batchnorm/add_1AddV2*batch_normalization_86/batchnorm/mul_1:z:0(batch_normalization_86/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_86/batchnorm/add_1v
Relu_1Relu*batch_normalization_86/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu_1?
dense_72/MatMul/ReadVariableOpReadVariableOp'dense_72_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_72/MatMul/ReadVariableOp?
dense_72/MatMulMatMulRelu_1:activations:0&dense_72/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_72/MatMul?
5batch_normalization_87/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_87/moments/mean/reduction_indices?
#batch_normalization_87/moments/meanMeandense_72/MatMul:product:0>batch_normalization_87/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_87/moments/mean?
+batch_normalization_87/moments/StopGradientStopGradient,batch_normalization_87/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_87/moments/StopGradient?
0batch_normalization_87/moments/SquaredDifferenceSquaredDifferencedense_72/MatMul:product:04batch_normalization_87/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????22
0batch_normalization_87/moments/SquaredDifference?
9batch_normalization_87/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_87/moments/variance/reduction_indices?
'batch_normalization_87/moments/varianceMean4batch_normalization_87/moments/SquaredDifference:z:0Bbatch_normalization_87/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_87/moments/variance?
&batch_normalization_87/moments/SqueezeSqueeze,batch_normalization_87/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_87/moments/Squeeze?
(batch_normalization_87/moments/Squeeze_1Squeeze0batch_normalization_87/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_87/moments/Squeeze_1?
,batch_normalization_87/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_87/AssignMovingAvg/decay?
+batch_normalization_87/AssignMovingAvg/CastCast5batch_normalization_87/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_87/AssignMovingAvg/Cast?
5batch_normalization_87/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_87_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_87/AssignMovingAvg/ReadVariableOp?
*batch_normalization_87/AssignMovingAvg/subSub=batch_normalization_87/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_87/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_87/AssignMovingAvg/sub?
*batch_normalization_87/AssignMovingAvg/mulMul.batch_normalization_87/AssignMovingAvg/sub:z:0/batch_normalization_87/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_87/AssignMovingAvg/mul?
&batch_normalization_87/AssignMovingAvgAssignSubVariableOp>batch_normalization_87_assignmovingavg_readvariableop_resource.batch_normalization_87/AssignMovingAvg/mul:z:06^batch_normalization_87/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_87/AssignMovingAvg?
.batch_normalization_87/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_87/AssignMovingAvg_1/decay?
-batch_normalization_87/AssignMovingAvg_1/CastCast7batch_normalization_87/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_87/AssignMovingAvg_1/Cast?
7batch_normalization_87/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_87_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_87/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_87/AssignMovingAvg_1/subSub?batch_normalization_87/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_87/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_87/AssignMovingAvg_1/sub?
,batch_normalization_87/AssignMovingAvg_1/mulMul0batch_normalization_87/AssignMovingAvg_1/sub:z:01batch_normalization_87/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_87/AssignMovingAvg_1/mul?
(batch_normalization_87/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_87_assignmovingavg_1_readvariableop_resource0batch_normalization_87/AssignMovingAvg_1/mul:z:08^batch_normalization_87/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_87/AssignMovingAvg_1?
&batch_normalization_87/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2(
&batch_normalization_87/batchnorm/add/y?
$batch_normalization_87/batchnorm/addAddV21batch_normalization_87/moments/Squeeze_1:output:0/batch_normalization_87/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_87/batchnorm/add?
&batch_normalization_87/batchnorm/RsqrtRsqrt(batch_normalization_87/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_87/batchnorm/Rsqrt?
3batch_normalization_87/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_87_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_87/batchnorm/mul/ReadVariableOp?
$batch_normalization_87/batchnorm/mulMul*batch_normalization_87/batchnorm/Rsqrt:y:0;batch_normalization_87/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_87/batchnorm/mul?
&batch_normalization_87/batchnorm/mul_1Muldense_72/MatMul:product:0(batch_normalization_87/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_87/batchnorm/mul_1?
&batch_normalization_87/batchnorm/mul_2Mul/batch_normalization_87/moments/Squeeze:output:0(batch_normalization_87/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_87/batchnorm/mul_2?
/batch_normalization_87/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_87_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_87/batchnorm/ReadVariableOp?
$batch_normalization_87/batchnorm/subSub7batch_normalization_87/batchnorm/ReadVariableOp:value:0*batch_normalization_87/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_87/batchnorm/sub?
&batch_normalization_87/batchnorm/add_1AddV2*batch_normalization_87/batchnorm/mul_1:z:0(batch_normalization_87/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_87/batchnorm/add_1v
Relu_2Relu*batch_normalization_87/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu_2?
dense_73/MatMul/ReadVariableOpReadVariableOp'dense_73_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_73/MatMul/ReadVariableOp?
dense_73/MatMulMatMulRelu_2:activations:0&dense_73/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_73/MatMul?
5batch_normalization_88/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_88/moments/mean/reduction_indices?
#batch_normalization_88/moments/meanMeandense_73/MatMul:product:0>batch_normalization_88/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_88/moments/mean?
+batch_normalization_88/moments/StopGradientStopGradient,batch_normalization_88/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_88/moments/StopGradient?
0batch_normalization_88/moments/SquaredDifferenceSquaredDifferencedense_73/MatMul:product:04batch_normalization_88/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????22
0batch_normalization_88/moments/SquaredDifference?
9batch_normalization_88/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_88/moments/variance/reduction_indices?
'batch_normalization_88/moments/varianceMean4batch_normalization_88/moments/SquaredDifference:z:0Bbatch_normalization_88/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_88/moments/variance?
&batch_normalization_88/moments/SqueezeSqueeze,batch_normalization_88/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_88/moments/Squeeze?
(batch_normalization_88/moments/Squeeze_1Squeeze0batch_normalization_88/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_88/moments/Squeeze_1?
,batch_normalization_88/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_88/AssignMovingAvg/decay?
+batch_normalization_88/AssignMovingAvg/CastCast5batch_normalization_88/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_88/AssignMovingAvg/Cast?
5batch_normalization_88/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_88_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_88/AssignMovingAvg/ReadVariableOp?
*batch_normalization_88/AssignMovingAvg/subSub=batch_normalization_88/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_88/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_88/AssignMovingAvg/sub?
*batch_normalization_88/AssignMovingAvg/mulMul.batch_normalization_88/AssignMovingAvg/sub:z:0/batch_normalization_88/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_88/AssignMovingAvg/mul?
&batch_normalization_88/AssignMovingAvgAssignSubVariableOp>batch_normalization_88_assignmovingavg_readvariableop_resource.batch_normalization_88/AssignMovingAvg/mul:z:06^batch_normalization_88/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_88/AssignMovingAvg?
.batch_normalization_88/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_88/AssignMovingAvg_1/decay?
-batch_normalization_88/AssignMovingAvg_1/CastCast7batch_normalization_88/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_88/AssignMovingAvg_1/Cast?
7batch_normalization_88/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_88_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_88/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_88/AssignMovingAvg_1/subSub?batch_normalization_88/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_88/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_88/AssignMovingAvg_1/sub?
,batch_normalization_88/AssignMovingAvg_1/mulMul0batch_normalization_88/AssignMovingAvg_1/sub:z:01batch_normalization_88/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_88/AssignMovingAvg_1/mul?
(batch_normalization_88/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_88_assignmovingavg_1_readvariableop_resource0batch_normalization_88/AssignMovingAvg_1/mul:z:08^batch_normalization_88/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_88/AssignMovingAvg_1?
&batch_normalization_88/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2(
&batch_normalization_88/batchnorm/add/y?
$batch_normalization_88/batchnorm/addAddV21batch_normalization_88/moments/Squeeze_1:output:0/batch_normalization_88/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_88/batchnorm/add?
&batch_normalization_88/batchnorm/RsqrtRsqrt(batch_normalization_88/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_88/batchnorm/Rsqrt?
3batch_normalization_88/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_88_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_88/batchnorm/mul/ReadVariableOp?
$batch_normalization_88/batchnorm/mulMul*batch_normalization_88/batchnorm/Rsqrt:y:0;batch_normalization_88/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_88/batchnorm/mul?
&batch_normalization_88/batchnorm/mul_1Muldense_73/MatMul:product:0(batch_normalization_88/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_88/batchnorm/mul_1?
&batch_normalization_88/batchnorm/mul_2Mul/batch_normalization_88/moments/Squeeze:output:0(batch_normalization_88/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_88/batchnorm/mul_2?
/batch_normalization_88/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_88_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_88/batchnorm/ReadVariableOp?
$batch_normalization_88/batchnorm/subSub7batch_normalization_88/batchnorm/ReadVariableOp:value:0*batch_normalization_88/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_88/batchnorm/sub?
&batch_normalization_88/batchnorm/add_1AddV2*batch_normalization_88/batchnorm/mul_1:z:0(batch_normalization_88/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_88/batchnorm/add_1v
Relu_3Relu*batch_normalization_88/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu_3?
dense_74/MatMul/ReadVariableOpReadVariableOp'dense_74_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_74/MatMul/ReadVariableOp?
dense_74/MatMulMatMulRelu_3:activations:0&dense_74/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_74/MatMul?
dense_74/BiasAdd/ReadVariableOpReadVariableOp(dense_74_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_74/BiasAdd/ReadVariableOp?
dense_74/BiasAddBiasAdddense_74/MatMul:product:0'dense_74/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_74/BiasAddt
IdentityIdentitydense_74/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp'^batch_normalization_84/AssignMovingAvg6^batch_normalization_84/AssignMovingAvg/ReadVariableOp)^batch_normalization_84/AssignMovingAvg_18^batch_normalization_84/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_84/batchnorm/ReadVariableOp4^batch_normalization_84/batchnorm/mul/ReadVariableOp'^batch_normalization_85/AssignMovingAvg6^batch_normalization_85/AssignMovingAvg/ReadVariableOp)^batch_normalization_85/AssignMovingAvg_18^batch_normalization_85/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_85/batchnorm/ReadVariableOp4^batch_normalization_85/batchnorm/mul/ReadVariableOp'^batch_normalization_86/AssignMovingAvg6^batch_normalization_86/AssignMovingAvg/ReadVariableOp)^batch_normalization_86/AssignMovingAvg_18^batch_normalization_86/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_86/batchnorm/ReadVariableOp4^batch_normalization_86/batchnorm/mul/ReadVariableOp'^batch_normalization_87/AssignMovingAvg6^batch_normalization_87/AssignMovingAvg/ReadVariableOp)^batch_normalization_87/AssignMovingAvg_18^batch_normalization_87/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_87/batchnorm/ReadVariableOp4^batch_normalization_87/batchnorm/mul/ReadVariableOp'^batch_normalization_88/AssignMovingAvg6^batch_normalization_88/AssignMovingAvg/ReadVariableOp)^batch_normalization_88/AssignMovingAvg_18^batch_normalization_88/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_88/batchnorm/ReadVariableOp4^batch_normalization_88/batchnorm/mul/ReadVariableOp^dense_70/MatMul/ReadVariableOp^dense_71/MatMul/ReadVariableOp^dense_72/MatMul/ReadVariableOp^dense_73/MatMul/ReadVariableOp ^dense_74/BiasAdd/ReadVariableOp^dense_74/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????
: : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_84/AssignMovingAvg&batch_normalization_84/AssignMovingAvg2n
5batch_normalization_84/AssignMovingAvg/ReadVariableOp5batch_normalization_84/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_84/AssignMovingAvg_1(batch_normalization_84/AssignMovingAvg_12r
7batch_normalization_84/AssignMovingAvg_1/ReadVariableOp7batch_normalization_84/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_84/batchnorm/ReadVariableOp/batch_normalization_84/batchnorm/ReadVariableOp2j
3batch_normalization_84/batchnorm/mul/ReadVariableOp3batch_normalization_84/batchnorm/mul/ReadVariableOp2P
&batch_normalization_85/AssignMovingAvg&batch_normalization_85/AssignMovingAvg2n
5batch_normalization_85/AssignMovingAvg/ReadVariableOp5batch_normalization_85/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_85/AssignMovingAvg_1(batch_normalization_85/AssignMovingAvg_12r
7batch_normalization_85/AssignMovingAvg_1/ReadVariableOp7batch_normalization_85/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_85/batchnorm/ReadVariableOp/batch_normalization_85/batchnorm/ReadVariableOp2j
3batch_normalization_85/batchnorm/mul/ReadVariableOp3batch_normalization_85/batchnorm/mul/ReadVariableOp2P
&batch_normalization_86/AssignMovingAvg&batch_normalization_86/AssignMovingAvg2n
5batch_normalization_86/AssignMovingAvg/ReadVariableOp5batch_normalization_86/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_86/AssignMovingAvg_1(batch_normalization_86/AssignMovingAvg_12r
7batch_normalization_86/AssignMovingAvg_1/ReadVariableOp7batch_normalization_86/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_86/batchnorm/ReadVariableOp/batch_normalization_86/batchnorm/ReadVariableOp2j
3batch_normalization_86/batchnorm/mul/ReadVariableOp3batch_normalization_86/batchnorm/mul/ReadVariableOp2P
&batch_normalization_87/AssignMovingAvg&batch_normalization_87/AssignMovingAvg2n
5batch_normalization_87/AssignMovingAvg/ReadVariableOp5batch_normalization_87/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_87/AssignMovingAvg_1(batch_normalization_87/AssignMovingAvg_12r
7batch_normalization_87/AssignMovingAvg_1/ReadVariableOp7batch_normalization_87/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_87/batchnorm/ReadVariableOp/batch_normalization_87/batchnorm/ReadVariableOp2j
3batch_normalization_87/batchnorm/mul/ReadVariableOp3batch_normalization_87/batchnorm/mul/ReadVariableOp2P
&batch_normalization_88/AssignMovingAvg&batch_normalization_88/AssignMovingAvg2n
5batch_normalization_88/AssignMovingAvg/ReadVariableOp5batch_normalization_88/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_88/AssignMovingAvg_1(batch_normalization_88/AssignMovingAvg_12r
7batch_normalization_88/AssignMovingAvg_1/ReadVariableOp7batch_normalization_88/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_88/batchnorm/ReadVariableOp/batch_normalization_88/batchnorm/ReadVariableOp2j
3batch_normalization_88/batchnorm/mul/ReadVariableOp3batch_normalization_88/batchnorm/mul/ReadVariableOp2@
dense_70/MatMul/ReadVariableOpdense_70/MatMul/ReadVariableOp2@
dense_71/MatMul/ReadVariableOpdense_71/MatMul/ReadVariableOp2@
dense_72/MatMul/ReadVariableOpdense_72/MatMul/ReadVariableOp2@
dense_73/MatMul/ReadVariableOpdense_73/MatMul/ReadVariableOp2B
dense_74/BiasAdd/ReadVariableOpdense_74/BiasAdd/ReadVariableOp2@
dense_74/MatMul/ReadVariableOpdense_74/MatMul/ReadVariableOp:J F
'
_output_shapes
:?????????


_user_specified_namex
?,
?
S__inference_batch_normalization_88_layer_call_and_return_conditional_losses_7061204

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
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
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
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
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg/Cast?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul?
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
?#<2
AssignMovingAvg_1/decay?
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg_1/Cast?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1k
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_86_layer_call_and_return_conditional_losses_7060810

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_dense_73_layer_call_and_return_conditional_losses_7061370

inputs0
matmul_readvariableop_resource:
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_87_layer_call_and_return_conditional_losses_7060976

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
9__inference_feed_forward_sub_net_14_layer_call_fn_7062166
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
identity??StatefulPartitionedCall?
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
:?????????
*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_feed_forward_sub_net_14_layer_call_and_return_conditional_losses_70616272
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
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
G:?????????
: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????

!
_user_specified_name	input_1
?
~
*__inference_dense_71_layer_call_fn_7063181

inputs
unknown:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_71_layer_call_and_return_conditional_losses_70613282
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?,
?
S__inference_batch_normalization_88_layer_call_and_return_conditional_losses_7063160

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
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
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
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
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg/Cast?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul?
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
?#<2
AssignMovingAvg_1/decay?
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg_1/Cast?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1k
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_dense_71_layer_call_and_return_conditional_losses_7061328

inputs0
matmul_readvariableop_resource:
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_84_layer_call_fn_7062763

inputs
unknown:

	unknown_0:

	unknown_1:

	unknown_2:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_84_layer_call_and_return_conditional_losses_70604782
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
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
:?????????
: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?,
?
S__inference_batch_normalization_85_layer_call_and_return_conditional_losses_7060706

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
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
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
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
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg/Cast?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul?
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
?#<2
AssignMovingAvg_1/decay?
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg_1/Cast?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1k
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
T__inference_feed_forward_sub_net_14_layer_call_and_return_conditional_losses_7062564
input_1F
8batch_normalization_84_batchnorm_readvariableop_resource:
J
<batch_normalization_84_batchnorm_mul_readvariableop_resource:
H
:batch_normalization_84_batchnorm_readvariableop_1_resource:
H
:batch_normalization_84_batchnorm_readvariableop_2_resource:
9
'dense_70_matmul_readvariableop_resource:
F
8batch_normalization_85_batchnorm_readvariableop_resource:J
<batch_normalization_85_batchnorm_mul_readvariableop_resource:H
:batch_normalization_85_batchnorm_readvariableop_1_resource:H
:batch_normalization_85_batchnorm_readvariableop_2_resource:9
'dense_71_matmul_readvariableop_resource:F
8batch_normalization_86_batchnorm_readvariableop_resource:J
<batch_normalization_86_batchnorm_mul_readvariableop_resource:H
:batch_normalization_86_batchnorm_readvariableop_1_resource:H
:batch_normalization_86_batchnorm_readvariableop_2_resource:9
'dense_72_matmul_readvariableop_resource:F
8batch_normalization_87_batchnorm_readvariableop_resource:J
<batch_normalization_87_batchnorm_mul_readvariableop_resource:H
:batch_normalization_87_batchnorm_readvariableop_1_resource:H
:batch_normalization_87_batchnorm_readvariableop_2_resource:9
'dense_73_matmul_readvariableop_resource:F
8batch_normalization_88_batchnorm_readvariableop_resource:J
<batch_normalization_88_batchnorm_mul_readvariableop_resource:H
:batch_normalization_88_batchnorm_readvariableop_1_resource:H
:batch_normalization_88_batchnorm_readvariableop_2_resource:9
'dense_74_matmul_readvariableop_resource:
6
(dense_74_biasadd_readvariableop_resource:

identity??/batch_normalization_84/batchnorm/ReadVariableOp?1batch_normalization_84/batchnorm/ReadVariableOp_1?1batch_normalization_84/batchnorm/ReadVariableOp_2?3batch_normalization_84/batchnorm/mul/ReadVariableOp?/batch_normalization_85/batchnorm/ReadVariableOp?1batch_normalization_85/batchnorm/ReadVariableOp_1?1batch_normalization_85/batchnorm/ReadVariableOp_2?3batch_normalization_85/batchnorm/mul/ReadVariableOp?/batch_normalization_86/batchnorm/ReadVariableOp?1batch_normalization_86/batchnorm/ReadVariableOp_1?1batch_normalization_86/batchnorm/ReadVariableOp_2?3batch_normalization_86/batchnorm/mul/ReadVariableOp?/batch_normalization_87/batchnorm/ReadVariableOp?1batch_normalization_87/batchnorm/ReadVariableOp_1?1batch_normalization_87/batchnorm/ReadVariableOp_2?3batch_normalization_87/batchnorm/mul/ReadVariableOp?/batch_normalization_88/batchnorm/ReadVariableOp?1batch_normalization_88/batchnorm/ReadVariableOp_1?1batch_normalization_88/batchnorm/ReadVariableOp_2?3batch_normalization_88/batchnorm/mul/ReadVariableOp?dense_70/MatMul/ReadVariableOp?dense_71/MatMul/ReadVariableOp?dense_72/MatMul/ReadVariableOp?dense_73/MatMul/ReadVariableOp?dense_74/BiasAdd/ReadVariableOp?dense_74/MatMul/ReadVariableOp?
/batch_normalization_84/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_84_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype021
/batch_normalization_84/batchnorm/ReadVariableOp?
&batch_normalization_84/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2(
&batch_normalization_84/batchnorm/add/y?
$batch_normalization_84/batchnorm/addAddV27batch_normalization_84/batchnorm/ReadVariableOp:value:0/batch_normalization_84/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2&
$batch_normalization_84/batchnorm/add?
&batch_normalization_84/batchnorm/RsqrtRsqrt(batch_normalization_84/batchnorm/add:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_84/batchnorm/Rsqrt?
3batch_normalization_84/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_84_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype025
3batch_normalization_84/batchnorm/mul/ReadVariableOp?
$batch_normalization_84/batchnorm/mulMul*batch_normalization_84/batchnorm/Rsqrt:y:0;batch_normalization_84/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2&
$batch_normalization_84/batchnorm/mul?
&batch_normalization_84/batchnorm/mul_1Mulinput_1(batch_normalization_84/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????
2(
&batch_normalization_84/batchnorm/mul_1?
1batch_normalization_84/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_84_batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype023
1batch_normalization_84/batchnorm/ReadVariableOp_1?
&batch_normalization_84/batchnorm/mul_2Mul9batch_normalization_84/batchnorm/ReadVariableOp_1:value:0(batch_normalization_84/batchnorm/mul:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_84/batchnorm/mul_2?
1batch_normalization_84/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_84_batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype023
1batch_normalization_84/batchnorm/ReadVariableOp_2?
$batch_normalization_84/batchnorm/subSub9batch_normalization_84/batchnorm/ReadVariableOp_2:value:0*batch_normalization_84/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2&
$batch_normalization_84/batchnorm/sub?
&batch_normalization_84/batchnorm/add_1AddV2*batch_normalization_84/batchnorm/mul_1:z:0(batch_normalization_84/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????
2(
&batch_normalization_84/batchnorm/add_1?
dense_70/MatMul/ReadVariableOpReadVariableOp'dense_70_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_70/MatMul/ReadVariableOp?
dense_70/MatMulMatMul*batch_normalization_84/batchnorm/add_1:z:0&dense_70/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_70/MatMul?
/batch_normalization_85/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_85_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_85/batchnorm/ReadVariableOp?
&batch_normalization_85/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2(
&batch_normalization_85/batchnorm/add/y?
$batch_normalization_85/batchnorm/addAddV27batch_normalization_85/batchnorm/ReadVariableOp:value:0/batch_normalization_85/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_85/batchnorm/add?
&batch_normalization_85/batchnorm/RsqrtRsqrt(batch_normalization_85/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_85/batchnorm/Rsqrt?
3batch_normalization_85/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_85_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_85/batchnorm/mul/ReadVariableOp?
$batch_normalization_85/batchnorm/mulMul*batch_normalization_85/batchnorm/Rsqrt:y:0;batch_normalization_85/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_85/batchnorm/mul?
&batch_normalization_85/batchnorm/mul_1Muldense_70/MatMul:product:0(batch_normalization_85/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_85/batchnorm/mul_1?
1batch_normalization_85/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_85_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_85/batchnorm/ReadVariableOp_1?
&batch_normalization_85/batchnorm/mul_2Mul9batch_normalization_85/batchnorm/ReadVariableOp_1:value:0(batch_normalization_85/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_85/batchnorm/mul_2?
1batch_normalization_85/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_85_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_85/batchnorm/ReadVariableOp_2?
$batch_normalization_85/batchnorm/subSub9batch_normalization_85/batchnorm/ReadVariableOp_2:value:0*batch_normalization_85/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_85/batchnorm/sub?
&batch_normalization_85/batchnorm/add_1AddV2*batch_normalization_85/batchnorm/mul_1:z:0(batch_normalization_85/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_85/batchnorm/add_1r
ReluRelu*batch_normalization_85/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu?
dense_71/MatMul/ReadVariableOpReadVariableOp'dense_71_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_71/MatMul/ReadVariableOp?
dense_71/MatMulMatMulRelu:activations:0&dense_71/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_71/MatMul?
/batch_normalization_86/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_86_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_86/batchnorm/ReadVariableOp?
&batch_normalization_86/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2(
&batch_normalization_86/batchnorm/add/y?
$batch_normalization_86/batchnorm/addAddV27batch_normalization_86/batchnorm/ReadVariableOp:value:0/batch_normalization_86/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_86/batchnorm/add?
&batch_normalization_86/batchnorm/RsqrtRsqrt(batch_normalization_86/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_86/batchnorm/Rsqrt?
3batch_normalization_86/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_86_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_86/batchnorm/mul/ReadVariableOp?
$batch_normalization_86/batchnorm/mulMul*batch_normalization_86/batchnorm/Rsqrt:y:0;batch_normalization_86/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_86/batchnorm/mul?
&batch_normalization_86/batchnorm/mul_1Muldense_71/MatMul:product:0(batch_normalization_86/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_86/batchnorm/mul_1?
1batch_normalization_86/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_86_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_86/batchnorm/ReadVariableOp_1?
&batch_normalization_86/batchnorm/mul_2Mul9batch_normalization_86/batchnorm/ReadVariableOp_1:value:0(batch_normalization_86/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_86/batchnorm/mul_2?
1batch_normalization_86/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_86_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_86/batchnorm/ReadVariableOp_2?
$batch_normalization_86/batchnorm/subSub9batch_normalization_86/batchnorm/ReadVariableOp_2:value:0*batch_normalization_86/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_86/batchnorm/sub?
&batch_normalization_86/batchnorm/add_1AddV2*batch_normalization_86/batchnorm/mul_1:z:0(batch_normalization_86/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_86/batchnorm/add_1v
Relu_1Relu*batch_normalization_86/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu_1?
dense_72/MatMul/ReadVariableOpReadVariableOp'dense_72_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_72/MatMul/ReadVariableOp?
dense_72/MatMulMatMulRelu_1:activations:0&dense_72/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_72/MatMul?
/batch_normalization_87/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_87_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_87/batchnorm/ReadVariableOp?
&batch_normalization_87/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2(
&batch_normalization_87/batchnorm/add/y?
$batch_normalization_87/batchnorm/addAddV27batch_normalization_87/batchnorm/ReadVariableOp:value:0/batch_normalization_87/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_87/batchnorm/add?
&batch_normalization_87/batchnorm/RsqrtRsqrt(batch_normalization_87/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_87/batchnorm/Rsqrt?
3batch_normalization_87/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_87_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_87/batchnorm/mul/ReadVariableOp?
$batch_normalization_87/batchnorm/mulMul*batch_normalization_87/batchnorm/Rsqrt:y:0;batch_normalization_87/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_87/batchnorm/mul?
&batch_normalization_87/batchnorm/mul_1Muldense_72/MatMul:product:0(batch_normalization_87/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_87/batchnorm/mul_1?
1batch_normalization_87/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_87_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_87/batchnorm/ReadVariableOp_1?
&batch_normalization_87/batchnorm/mul_2Mul9batch_normalization_87/batchnorm/ReadVariableOp_1:value:0(batch_normalization_87/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_87/batchnorm/mul_2?
1batch_normalization_87/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_87_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_87/batchnorm/ReadVariableOp_2?
$batch_normalization_87/batchnorm/subSub9batch_normalization_87/batchnorm/ReadVariableOp_2:value:0*batch_normalization_87/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_87/batchnorm/sub?
&batch_normalization_87/batchnorm/add_1AddV2*batch_normalization_87/batchnorm/mul_1:z:0(batch_normalization_87/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_87/batchnorm/add_1v
Relu_2Relu*batch_normalization_87/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu_2?
dense_73/MatMul/ReadVariableOpReadVariableOp'dense_73_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_73/MatMul/ReadVariableOp?
dense_73/MatMulMatMulRelu_2:activations:0&dense_73/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_73/MatMul?
/batch_normalization_88/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_88_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_88/batchnorm/ReadVariableOp?
&batch_normalization_88/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2(
&batch_normalization_88/batchnorm/add/y?
$batch_normalization_88/batchnorm/addAddV27batch_normalization_88/batchnorm/ReadVariableOp:value:0/batch_normalization_88/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_88/batchnorm/add?
&batch_normalization_88/batchnorm/RsqrtRsqrt(batch_normalization_88/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_88/batchnorm/Rsqrt?
3batch_normalization_88/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_88_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_88/batchnorm/mul/ReadVariableOp?
$batch_normalization_88/batchnorm/mulMul*batch_normalization_88/batchnorm/Rsqrt:y:0;batch_normalization_88/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_88/batchnorm/mul?
&batch_normalization_88/batchnorm/mul_1Muldense_73/MatMul:product:0(batch_normalization_88/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_88/batchnorm/mul_1?
1batch_normalization_88/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_88_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_88/batchnorm/ReadVariableOp_1?
&batch_normalization_88/batchnorm/mul_2Mul9batch_normalization_88/batchnorm/ReadVariableOp_1:value:0(batch_normalization_88/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_88/batchnorm/mul_2?
1batch_normalization_88/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_88_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_88/batchnorm/ReadVariableOp_2?
$batch_normalization_88/batchnorm/subSub9batch_normalization_88/batchnorm/ReadVariableOp_2:value:0*batch_normalization_88/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_88/batchnorm/sub?
&batch_normalization_88/batchnorm/add_1AddV2*batch_normalization_88/batchnorm/mul_1:z:0(batch_normalization_88/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_88/batchnorm/add_1v
Relu_3Relu*batch_normalization_88/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu_3?
dense_74/MatMul/ReadVariableOpReadVariableOp'dense_74_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_74/MatMul/ReadVariableOp?
dense_74/MatMulMatMulRelu_3:activations:0&dense_74/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_74/MatMul?
dense_74/BiasAdd/ReadVariableOpReadVariableOp(dense_74_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_74/BiasAdd/ReadVariableOp?
dense_74/BiasAddBiasAdddense_74/MatMul:product:0'dense_74/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_74/BiasAddt
IdentityIdentitydense_74/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?

NoOpNoOp0^batch_normalization_84/batchnorm/ReadVariableOp2^batch_normalization_84/batchnorm/ReadVariableOp_12^batch_normalization_84/batchnorm/ReadVariableOp_24^batch_normalization_84/batchnorm/mul/ReadVariableOp0^batch_normalization_85/batchnorm/ReadVariableOp2^batch_normalization_85/batchnorm/ReadVariableOp_12^batch_normalization_85/batchnorm/ReadVariableOp_24^batch_normalization_85/batchnorm/mul/ReadVariableOp0^batch_normalization_86/batchnorm/ReadVariableOp2^batch_normalization_86/batchnorm/ReadVariableOp_12^batch_normalization_86/batchnorm/ReadVariableOp_24^batch_normalization_86/batchnorm/mul/ReadVariableOp0^batch_normalization_87/batchnorm/ReadVariableOp2^batch_normalization_87/batchnorm/ReadVariableOp_12^batch_normalization_87/batchnorm/ReadVariableOp_24^batch_normalization_87/batchnorm/mul/ReadVariableOp0^batch_normalization_88/batchnorm/ReadVariableOp2^batch_normalization_88/batchnorm/ReadVariableOp_12^batch_normalization_88/batchnorm/ReadVariableOp_24^batch_normalization_88/batchnorm/mul/ReadVariableOp^dense_70/MatMul/ReadVariableOp^dense_71/MatMul/ReadVariableOp^dense_72/MatMul/ReadVariableOp^dense_73/MatMul/ReadVariableOp ^dense_74/BiasAdd/ReadVariableOp^dense_74/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????
: : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_84/batchnorm/ReadVariableOp/batch_normalization_84/batchnorm/ReadVariableOp2f
1batch_normalization_84/batchnorm/ReadVariableOp_11batch_normalization_84/batchnorm/ReadVariableOp_12f
1batch_normalization_84/batchnorm/ReadVariableOp_21batch_normalization_84/batchnorm/ReadVariableOp_22j
3batch_normalization_84/batchnorm/mul/ReadVariableOp3batch_normalization_84/batchnorm/mul/ReadVariableOp2b
/batch_normalization_85/batchnorm/ReadVariableOp/batch_normalization_85/batchnorm/ReadVariableOp2f
1batch_normalization_85/batchnorm/ReadVariableOp_11batch_normalization_85/batchnorm/ReadVariableOp_12f
1batch_normalization_85/batchnorm/ReadVariableOp_21batch_normalization_85/batchnorm/ReadVariableOp_22j
3batch_normalization_85/batchnorm/mul/ReadVariableOp3batch_normalization_85/batchnorm/mul/ReadVariableOp2b
/batch_normalization_86/batchnorm/ReadVariableOp/batch_normalization_86/batchnorm/ReadVariableOp2f
1batch_normalization_86/batchnorm/ReadVariableOp_11batch_normalization_86/batchnorm/ReadVariableOp_12f
1batch_normalization_86/batchnorm/ReadVariableOp_21batch_normalization_86/batchnorm/ReadVariableOp_22j
3batch_normalization_86/batchnorm/mul/ReadVariableOp3batch_normalization_86/batchnorm/mul/ReadVariableOp2b
/batch_normalization_87/batchnorm/ReadVariableOp/batch_normalization_87/batchnorm/ReadVariableOp2f
1batch_normalization_87/batchnorm/ReadVariableOp_11batch_normalization_87/batchnorm/ReadVariableOp_12f
1batch_normalization_87/batchnorm/ReadVariableOp_21batch_normalization_87/batchnorm/ReadVariableOp_22j
3batch_normalization_87/batchnorm/mul/ReadVariableOp3batch_normalization_87/batchnorm/mul/ReadVariableOp2b
/batch_normalization_88/batchnorm/ReadVariableOp/batch_normalization_88/batchnorm/ReadVariableOp2f
1batch_normalization_88/batchnorm/ReadVariableOp_11batch_normalization_88/batchnorm/ReadVariableOp_12f
1batch_normalization_88/batchnorm/ReadVariableOp_21batch_normalization_88/batchnorm/ReadVariableOp_22j
3batch_normalization_88/batchnorm/mul/ReadVariableOp3batch_normalization_88/batchnorm/mul/ReadVariableOp2@
dense_70/MatMul/ReadVariableOpdense_70/MatMul/ReadVariableOp2@
dense_71/MatMul/ReadVariableOpdense_71/MatMul/ReadVariableOp2@
dense_72/MatMul/ReadVariableOpdense_72/MatMul/ReadVariableOp2@
dense_73/MatMul/ReadVariableOpdense_73/MatMul/ReadVariableOp2B
dense_74/BiasAdd/ReadVariableOpdense_74/BiasAdd/ReadVariableOp2@
dense_74/MatMul/ReadVariableOpdense_74/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????

!
_user_specified_name	input_1
?

?
E__inference_dense_74_layer_call_and_return_conditional_losses_7063235

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
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
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
9__inference_feed_forward_sub_net_14_layer_call_fn_7062052
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
identity??StatefulPartitionedCall?
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
:?????????
*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_feed_forward_sub_net_14_layer_call_and_return_conditional_losses_70614012
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
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
G:?????????
: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:?????????


_user_specified_namex
?
~
*__inference_dense_73_layer_call_fn_7063209

inputs
unknown:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_73_layer_call_and_return_conditional_losses_70613702
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
E__inference_dense_74_layer_call_and_return_conditional_losses_7061394

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
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
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
9__inference_feed_forward_sub_net_14_layer_call_fn_7062109
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
identity??StatefulPartitionedCall?
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
:?????????
*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_feed_forward_sub_net_14_layer_call_and_return_conditional_losses_70616272
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
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
G:?????????
: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:?????????


_user_specified_namex
?,
?
S__inference_batch_normalization_86_layer_call_and_return_conditional_losses_7062996

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
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
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
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
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg/Cast?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul?
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
?#<2
AssignMovingAvg_1/decay?
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg_1/Cast?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1k
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
%__inference_signature_wrapper_7061938
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
identity??StatefulPartitionedCall?
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
:?????????
*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_70604542
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
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
G:?????????
: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????

!
_user_specified_name	input_1
?
?
E__inference_dense_72_layer_call_and_return_conditional_losses_7063202

inputs0
matmul_readvariableop_resource:
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_88_layer_call_and_return_conditional_losses_7061142

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
~
*__inference_dense_72_layer_call_fn_7063195

inputs
unknown:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_72_layer_call_and_return_conditional_losses_70613492
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_dense_73_layer_call_and_return_conditional_losses_7063216

inputs0
matmul_readvariableop_resource:
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
T__inference_feed_forward_sub_net_14_layer_call_and_return_conditional_losses_7062272
xF
8batch_normalization_84_batchnorm_readvariableop_resource:
J
<batch_normalization_84_batchnorm_mul_readvariableop_resource:
H
:batch_normalization_84_batchnorm_readvariableop_1_resource:
H
:batch_normalization_84_batchnorm_readvariableop_2_resource:
9
'dense_70_matmul_readvariableop_resource:
F
8batch_normalization_85_batchnorm_readvariableop_resource:J
<batch_normalization_85_batchnorm_mul_readvariableop_resource:H
:batch_normalization_85_batchnorm_readvariableop_1_resource:H
:batch_normalization_85_batchnorm_readvariableop_2_resource:9
'dense_71_matmul_readvariableop_resource:F
8batch_normalization_86_batchnorm_readvariableop_resource:J
<batch_normalization_86_batchnorm_mul_readvariableop_resource:H
:batch_normalization_86_batchnorm_readvariableop_1_resource:H
:batch_normalization_86_batchnorm_readvariableop_2_resource:9
'dense_72_matmul_readvariableop_resource:F
8batch_normalization_87_batchnorm_readvariableop_resource:J
<batch_normalization_87_batchnorm_mul_readvariableop_resource:H
:batch_normalization_87_batchnorm_readvariableop_1_resource:H
:batch_normalization_87_batchnorm_readvariableop_2_resource:9
'dense_73_matmul_readvariableop_resource:F
8batch_normalization_88_batchnorm_readvariableop_resource:J
<batch_normalization_88_batchnorm_mul_readvariableop_resource:H
:batch_normalization_88_batchnorm_readvariableop_1_resource:H
:batch_normalization_88_batchnorm_readvariableop_2_resource:9
'dense_74_matmul_readvariableop_resource:
6
(dense_74_biasadd_readvariableop_resource:

identity??/batch_normalization_84/batchnorm/ReadVariableOp?1batch_normalization_84/batchnorm/ReadVariableOp_1?1batch_normalization_84/batchnorm/ReadVariableOp_2?3batch_normalization_84/batchnorm/mul/ReadVariableOp?/batch_normalization_85/batchnorm/ReadVariableOp?1batch_normalization_85/batchnorm/ReadVariableOp_1?1batch_normalization_85/batchnorm/ReadVariableOp_2?3batch_normalization_85/batchnorm/mul/ReadVariableOp?/batch_normalization_86/batchnorm/ReadVariableOp?1batch_normalization_86/batchnorm/ReadVariableOp_1?1batch_normalization_86/batchnorm/ReadVariableOp_2?3batch_normalization_86/batchnorm/mul/ReadVariableOp?/batch_normalization_87/batchnorm/ReadVariableOp?1batch_normalization_87/batchnorm/ReadVariableOp_1?1batch_normalization_87/batchnorm/ReadVariableOp_2?3batch_normalization_87/batchnorm/mul/ReadVariableOp?/batch_normalization_88/batchnorm/ReadVariableOp?1batch_normalization_88/batchnorm/ReadVariableOp_1?1batch_normalization_88/batchnorm/ReadVariableOp_2?3batch_normalization_88/batchnorm/mul/ReadVariableOp?dense_70/MatMul/ReadVariableOp?dense_71/MatMul/ReadVariableOp?dense_72/MatMul/ReadVariableOp?dense_73/MatMul/ReadVariableOp?dense_74/BiasAdd/ReadVariableOp?dense_74/MatMul/ReadVariableOp?
/batch_normalization_84/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_84_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype021
/batch_normalization_84/batchnorm/ReadVariableOp?
&batch_normalization_84/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2(
&batch_normalization_84/batchnorm/add/y?
$batch_normalization_84/batchnorm/addAddV27batch_normalization_84/batchnorm/ReadVariableOp:value:0/batch_normalization_84/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2&
$batch_normalization_84/batchnorm/add?
&batch_normalization_84/batchnorm/RsqrtRsqrt(batch_normalization_84/batchnorm/add:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_84/batchnorm/Rsqrt?
3batch_normalization_84/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_84_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype025
3batch_normalization_84/batchnorm/mul/ReadVariableOp?
$batch_normalization_84/batchnorm/mulMul*batch_normalization_84/batchnorm/Rsqrt:y:0;batch_normalization_84/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2&
$batch_normalization_84/batchnorm/mul?
&batch_normalization_84/batchnorm/mul_1Mulx(batch_normalization_84/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????
2(
&batch_normalization_84/batchnorm/mul_1?
1batch_normalization_84/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_84_batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype023
1batch_normalization_84/batchnorm/ReadVariableOp_1?
&batch_normalization_84/batchnorm/mul_2Mul9batch_normalization_84/batchnorm/ReadVariableOp_1:value:0(batch_normalization_84/batchnorm/mul:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_84/batchnorm/mul_2?
1batch_normalization_84/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_84_batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype023
1batch_normalization_84/batchnorm/ReadVariableOp_2?
$batch_normalization_84/batchnorm/subSub9batch_normalization_84/batchnorm/ReadVariableOp_2:value:0*batch_normalization_84/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2&
$batch_normalization_84/batchnorm/sub?
&batch_normalization_84/batchnorm/add_1AddV2*batch_normalization_84/batchnorm/mul_1:z:0(batch_normalization_84/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????
2(
&batch_normalization_84/batchnorm/add_1?
dense_70/MatMul/ReadVariableOpReadVariableOp'dense_70_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_70/MatMul/ReadVariableOp?
dense_70/MatMulMatMul*batch_normalization_84/batchnorm/add_1:z:0&dense_70/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_70/MatMul?
/batch_normalization_85/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_85_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_85/batchnorm/ReadVariableOp?
&batch_normalization_85/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2(
&batch_normalization_85/batchnorm/add/y?
$batch_normalization_85/batchnorm/addAddV27batch_normalization_85/batchnorm/ReadVariableOp:value:0/batch_normalization_85/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_85/batchnorm/add?
&batch_normalization_85/batchnorm/RsqrtRsqrt(batch_normalization_85/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_85/batchnorm/Rsqrt?
3batch_normalization_85/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_85_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_85/batchnorm/mul/ReadVariableOp?
$batch_normalization_85/batchnorm/mulMul*batch_normalization_85/batchnorm/Rsqrt:y:0;batch_normalization_85/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_85/batchnorm/mul?
&batch_normalization_85/batchnorm/mul_1Muldense_70/MatMul:product:0(batch_normalization_85/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_85/batchnorm/mul_1?
1batch_normalization_85/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_85_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_85/batchnorm/ReadVariableOp_1?
&batch_normalization_85/batchnorm/mul_2Mul9batch_normalization_85/batchnorm/ReadVariableOp_1:value:0(batch_normalization_85/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_85/batchnorm/mul_2?
1batch_normalization_85/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_85_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_85/batchnorm/ReadVariableOp_2?
$batch_normalization_85/batchnorm/subSub9batch_normalization_85/batchnorm/ReadVariableOp_2:value:0*batch_normalization_85/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_85/batchnorm/sub?
&batch_normalization_85/batchnorm/add_1AddV2*batch_normalization_85/batchnorm/mul_1:z:0(batch_normalization_85/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_85/batchnorm/add_1r
ReluRelu*batch_normalization_85/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu?
dense_71/MatMul/ReadVariableOpReadVariableOp'dense_71_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_71/MatMul/ReadVariableOp?
dense_71/MatMulMatMulRelu:activations:0&dense_71/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_71/MatMul?
/batch_normalization_86/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_86_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_86/batchnorm/ReadVariableOp?
&batch_normalization_86/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2(
&batch_normalization_86/batchnorm/add/y?
$batch_normalization_86/batchnorm/addAddV27batch_normalization_86/batchnorm/ReadVariableOp:value:0/batch_normalization_86/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_86/batchnorm/add?
&batch_normalization_86/batchnorm/RsqrtRsqrt(batch_normalization_86/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_86/batchnorm/Rsqrt?
3batch_normalization_86/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_86_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_86/batchnorm/mul/ReadVariableOp?
$batch_normalization_86/batchnorm/mulMul*batch_normalization_86/batchnorm/Rsqrt:y:0;batch_normalization_86/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_86/batchnorm/mul?
&batch_normalization_86/batchnorm/mul_1Muldense_71/MatMul:product:0(batch_normalization_86/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_86/batchnorm/mul_1?
1batch_normalization_86/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_86_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_86/batchnorm/ReadVariableOp_1?
&batch_normalization_86/batchnorm/mul_2Mul9batch_normalization_86/batchnorm/ReadVariableOp_1:value:0(batch_normalization_86/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_86/batchnorm/mul_2?
1batch_normalization_86/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_86_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_86/batchnorm/ReadVariableOp_2?
$batch_normalization_86/batchnorm/subSub9batch_normalization_86/batchnorm/ReadVariableOp_2:value:0*batch_normalization_86/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_86/batchnorm/sub?
&batch_normalization_86/batchnorm/add_1AddV2*batch_normalization_86/batchnorm/mul_1:z:0(batch_normalization_86/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_86/batchnorm/add_1v
Relu_1Relu*batch_normalization_86/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu_1?
dense_72/MatMul/ReadVariableOpReadVariableOp'dense_72_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_72/MatMul/ReadVariableOp?
dense_72/MatMulMatMulRelu_1:activations:0&dense_72/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_72/MatMul?
/batch_normalization_87/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_87_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_87/batchnorm/ReadVariableOp?
&batch_normalization_87/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2(
&batch_normalization_87/batchnorm/add/y?
$batch_normalization_87/batchnorm/addAddV27batch_normalization_87/batchnorm/ReadVariableOp:value:0/batch_normalization_87/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_87/batchnorm/add?
&batch_normalization_87/batchnorm/RsqrtRsqrt(batch_normalization_87/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_87/batchnorm/Rsqrt?
3batch_normalization_87/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_87_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_87/batchnorm/mul/ReadVariableOp?
$batch_normalization_87/batchnorm/mulMul*batch_normalization_87/batchnorm/Rsqrt:y:0;batch_normalization_87/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_87/batchnorm/mul?
&batch_normalization_87/batchnorm/mul_1Muldense_72/MatMul:product:0(batch_normalization_87/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_87/batchnorm/mul_1?
1batch_normalization_87/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_87_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_87/batchnorm/ReadVariableOp_1?
&batch_normalization_87/batchnorm/mul_2Mul9batch_normalization_87/batchnorm/ReadVariableOp_1:value:0(batch_normalization_87/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_87/batchnorm/mul_2?
1batch_normalization_87/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_87_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_87/batchnorm/ReadVariableOp_2?
$batch_normalization_87/batchnorm/subSub9batch_normalization_87/batchnorm/ReadVariableOp_2:value:0*batch_normalization_87/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_87/batchnorm/sub?
&batch_normalization_87/batchnorm/add_1AddV2*batch_normalization_87/batchnorm/mul_1:z:0(batch_normalization_87/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_87/batchnorm/add_1v
Relu_2Relu*batch_normalization_87/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu_2?
dense_73/MatMul/ReadVariableOpReadVariableOp'dense_73_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_73/MatMul/ReadVariableOp?
dense_73/MatMulMatMulRelu_2:activations:0&dense_73/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_73/MatMul?
/batch_normalization_88/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_88_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_88/batchnorm/ReadVariableOp?
&batch_normalization_88/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2(
&batch_normalization_88/batchnorm/add/y?
$batch_normalization_88/batchnorm/addAddV27batch_normalization_88/batchnorm/ReadVariableOp:value:0/batch_normalization_88/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_88/batchnorm/add?
&batch_normalization_88/batchnorm/RsqrtRsqrt(batch_normalization_88/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_88/batchnorm/Rsqrt?
3batch_normalization_88/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_88_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_88/batchnorm/mul/ReadVariableOp?
$batch_normalization_88/batchnorm/mulMul*batch_normalization_88/batchnorm/Rsqrt:y:0;batch_normalization_88/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_88/batchnorm/mul?
&batch_normalization_88/batchnorm/mul_1Muldense_73/MatMul:product:0(batch_normalization_88/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_88/batchnorm/mul_1?
1batch_normalization_88/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_88_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_88/batchnorm/ReadVariableOp_1?
&batch_normalization_88/batchnorm/mul_2Mul9batch_normalization_88/batchnorm/ReadVariableOp_1:value:0(batch_normalization_88/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_88/batchnorm/mul_2?
1batch_normalization_88/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_88_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_88/batchnorm/ReadVariableOp_2?
$batch_normalization_88/batchnorm/subSub9batch_normalization_88/batchnorm/ReadVariableOp_2:value:0*batch_normalization_88/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_88/batchnorm/sub?
&batch_normalization_88/batchnorm/add_1AddV2*batch_normalization_88/batchnorm/mul_1:z:0(batch_normalization_88/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_88/batchnorm/add_1v
Relu_3Relu*batch_normalization_88/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu_3?
dense_74/MatMul/ReadVariableOpReadVariableOp'dense_74_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_74/MatMul/ReadVariableOp?
dense_74/MatMulMatMulRelu_3:activations:0&dense_74/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_74/MatMul?
dense_74/BiasAdd/ReadVariableOpReadVariableOp(dense_74_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_74/BiasAdd/ReadVariableOp?
dense_74/BiasAddBiasAdddense_74/MatMul:product:0'dense_74/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_74/BiasAddt
IdentityIdentitydense_74/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?

NoOpNoOp0^batch_normalization_84/batchnorm/ReadVariableOp2^batch_normalization_84/batchnorm/ReadVariableOp_12^batch_normalization_84/batchnorm/ReadVariableOp_24^batch_normalization_84/batchnorm/mul/ReadVariableOp0^batch_normalization_85/batchnorm/ReadVariableOp2^batch_normalization_85/batchnorm/ReadVariableOp_12^batch_normalization_85/batchnorm/ReadVariableOp_24^batch_normalization_85/batchnorm/mul/ReadVariableOp0^batch_normalization_86/batchnorm/ReadVariableOp2^batch_normalization_86/batchnorm/ReadVariableOp_12^batch_normalization_86/batchnorm/ReadVariableOp_24^batch_normalization_86/batchnorm/mul/ReadVariableOp0^batch_normalization_87/batchnorm/ReadVariableOp2^batch_normalization_87/batchnorm/ReadVariableOp_12^batch_normalization_87/batchnorm/ReadVariableOp_24^batch_normalization_87/batchnorm/mul/ReadVariableOp0^batch_normalization_88/batchnorm/ReadVariableOp2^batch_normalization_88/batchnorm/ReadVariableOp_12^batch_normalization_88/batchnorm/ReadVariableOp_24^batch_normalization_88/batchnorm/mul/ReadVariableOp^dense_70/MatMul/ReadVariableOp^dense_71/MatMul/ReadVariableOp^dense_72/MatMul/ReadVariableOp^dense_73/MatMul/ReadVariableOp ^dense_74/BiasAdd/ReadVariableOp^dense_74/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????
: : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_84/batchnorm/ReadVariableOp/batch_normalization_84/batchnorm/ReadVariableOp2f
1batch_normalization_84/batchnorm/ReadVariableOp_11batch_normalization_84/batchnorm/ReadVariableOp_12f
1batch_normalization_84/batchnorm/ReadVariableOp_21batch_normalization_84/batchnorm/ReadVariableOp_22j
3batch_normalization_84/batchnorm/mul/ReadVariableOp3batch_normalization_84/batchnorm/mul/ReadVariableOp2b
/batch_normalization_85/batchnorm/ReadVariableOp/batch_normalization_85/batchnorm/ReadVariableOp2f
1batch_normalization_85/batchnorm/ReadVariableOp_11batch_normalization_85/batchnorm/ReadVariableOp_12f
1batch_normalization_85/batchnorm/ReadVariableOp_21batch_normalization_85/batchnorm/ReadVariableOp_22j
3batch_normalization_85/batchnorm/mul/ReadVariableOp3batch_normalization_85/batchnorm/mul/ReadVariableOp2b
/batch_normalization_86/batchnorm/ReadVariableOp/batch_normalization_86/batchnorm/ReadVariableOp2f
1batch_normalization_86/batchnorm/ReadVariableOp_11batch_normalization_86/batchnorm/ReadVariableOp_12f
1batch_normalization_86/batchnorm/ReadVariableOp_21batch_normalization_86/batchnorm/ReadVariableOp_22j
3batch_normalization_86/batchnorm/mul/ReadVariableOp3batch_normalization_86/batchnorm/mul/ReadVariableOp2b
/batch_normalization_87/batchnorm/ReadVariableOp/batch_normalization_87/batchnorm/ReadVariableOp2f
1batch_normalization_87/batchnorm/ReadVariableOp_11batch_normalization_87/batchnorm/ReadVariableOp_12f
1batch_normalization_87/batchnorm/ReadVariableOp_21batch_normalization_87/batchnorm/ReadVariableOp_22j
3batch_normalization_87/batchnorm/mul/ReadVariableOp3batch_normalization_87/batchnorm/mul/ReadVariableOp2b
/batch_normalization_88/batchnorm/ReadVariableOp/batch_normalization_88/batchnorm/ReadVariableOp2f
1batch_normalization_88/batchnorm/ReadVariableOp_11batch_normalization_88/batchnorm/ReadVariableOp_12f
1batch_normalization_88/batchnorm/ReadVariableOp_21batch_normalization_88/batchnorm/ReadVariableOp_22j
3batch_normalization_88/batchnorm/mul/ReadVariableOp3batch_normalization_88/batchnorm/mul/ReadVariableOp2@
dense_70/MatMul/ReadVariableOpdense_70/MatMul/ReadVariableOp2@
dense_71/MatMul/ReadVariableOpdense_71/MatMul/ReadVariableOp2@
dense_72/MatMul/ReadVariableOpdense_72/MatMul/ReadVariableOp2@
dense_73/MatMul/ReadVariableOpdense_73/MatMul/ReadVariableOp2B
dense_74/BiasAdd/ReadVariableOpdense_74/BiasAdd/ReadVariableOp2@
dense_74/MatMul/ReadVariableOpdense_74/MatMul/ReadVariableOp:J F
'
_output_shapes
:?????????


_user_specified_namex
?,
?
S__inference_batch_normalization_86_layer_call_and_return_conditional_losses_7060872

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
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
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
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
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg/Cast?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul?
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
?#<2
AssignMovingAvg_1/decay?
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg_1/Cast?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1k
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_dense_70_layer_call_and_return_conditional_losses_7063174

inputs0
matmul_readvariableop_resource:

identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????
: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_84_layer_call_and_return_conditional_losses_7062796

inputs/
!batchnorm_readvariableop_resource:
3
%batchnorm_mul_readvariableop_resource:
1
#batchnorm_readvariableop_1_resource:
1
#batchnorm_readvariableop_2_resource:

identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
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
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????
2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:
2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????
2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
??
?"
"__inference__wrapped_model_7060454
input_1^
Pfeed_forward_sub_net_14_batch_normalization_84_batchnorm_readvariableop_resource:
b
Tfeed_forward_sub_net_14_batch_normalization_84_batchnorm_mul_readvariableop_resource:
`
Rfeed_forward_sub_net_14_batch_normalization_84_batchnorm_readvariableop_1_resource:
`
Rfeed_forward_sub_net_14_batch_normalization_84_batchnorm_readvariableop_2_resource:
Q
?feed_forward_sub_net_14_dense_70_matmul_readvariableop_resource:
^
Pfeed_forward_sub_net_14_batch_normalization_85_batchnorm_readvariableop_resource:b
Tfeed_forward_sub_net_14_batch_normalization_85_batchnorm_mul_readvariableop_resource:`
Rfeed_forward_sub_net_14_batch_normalization_85_batchnorm_readvariableop_1_resource:`
Rfeed_forward_sub_net_14_batch_normalization_85_batchnorm_readvariableop_2_resource:Q
?feed_forward_sub_net_14_dense_71_matmul_readvariableop_resource:^
Pfeed_forward_sub_net_14_batch_normalization_86_batchnorm_readvariableop_resource:b
Tfeed_forward_sub_net_14_batch_normalization_86_batchnorm_mul_readvariableop_resource:`
Rfeed_forward_sub_net_14_batch_normalization_86_batchnorm_readvariableop_1_resource:`
Rfeed_forward_sub_net_14_batch_normalization_86_batchnorm_readvariableop_2_resource:Q
?feed_forward_sub_net_14_dense_72_matmul_readvariableop_resource:^
Pfeed_forward_sub_net_14_batch_normalization_87_batchnorm_readvariableop_resource:b
Tfeed_forward_sub_net_14_batch_normalization_87_batchnorm_mul_readvariableop_resource:`
Rfeed_forward_sub_net_14_batch_normalization_87_batchnorm_readvariableop_1_resource:`
Rfeed_forward_sub_net_14_batch_normalization_87_batchnorm_readvariableop_2_resource:Q
?feed_forward_sub_net_14_dense_73_matmul_readvariableop_resource:^
Pfeed_forward_sub_net_14_batch_normalization_88_batchnorm_readvariableop_resource:b
Tfeed_forward_sub_net_14_batch_normalization_88_batchnorm_mul_readvariableop_resource:`
Rfeed_forward_sub_net_14_batch_normalization_88_batchnorm_readvariableop_1_resource:`
Rfeed_forward_sub_net_14_batch_normalization_88_batchnorm_readvariableop_2_resource:Q
?feed_forward_sub_net_14_dense_74_matmul_readvariableop_resource:
N
@feed_forward_sub_net_14_dense_74_biasadd_readvariableop_resource:

identity??Gfeed_forward_sub_net_14/batch_normalization_84/batchnorm/ReadVariableOp?Ifeed_forward_sub_net_14/batch_normalization_84/batchnorm/ReadVariableOp_1?Ifeed_forward_sub_net_14/batch_normalization_84/batchnorm/ReadVariableOp_2?Kfeed_forward_sub_net_14/batch_normalization_84/batchnorm/mul/ReadVariableOp?Gfeed_forward_sub_net_14/batch_normalization_85/batchnorm/ReadVariableOp?Ifeed_forward_sub_net_14/batch_normalization_85/batchnorm/ReadVariableOp_1?Ifeed_forward_sub_net_14/batch_normalization_85/batchnorm/ReadVariableOp_2?Kfeed_forward_sub_net_14/batch_normalization_85/batchnorm/mul/ReadVariableOp?Gfeed_forward_sub_net_14/batch_normalization_86/batchnorm/ReadVariableOp?Ifeed_forward_sub_net_14/batch_normalization_86/batchnorm/ReadVariableOp_1?Ifeed_forward_sub_net_14/batch_normalization_86/batchnorm/ReadVariableOp_2?Kfeed_forward_sub_net_14/batch_normalization_86/batchnorm/mul/ReadVariableOp?Gfeed_forward_sub_net_14/batch_normalization_87/batchnorm/ReadVariableOp?Ifeed_forward_sub_net_14/batch_normalization_87/batchnorm/ReadVariableOp_1?Ifeed_forward_sub_net_14/batch_normalization_87/batchnorm/ReadVariableOp_2?Kfeed_forward_sub_net_14/batch_normalization_87/batchnorm/mul/ReadVariableOp?Gfeed_forward_sub_net_14/batch_normalization_88/batchnorm/ReadVariableOp?Ifeed_forward_sub_net_14/batch_normalization_88/batchnorm/ReadVariableOp_1?Ifeed_forward_sub_net_14/batch_normalization_88/batchnorm/ReadVariableOp_2?Kfeed_forward_sub_net_14/batch_normalization_88/batchnorm/mul/ReadVariableOp?6feed_forward_sub_net_14/dense_70/MatMul/ReadVariableOp?6feed_forward_sub_net_14/dense_71/MatMul/ReadVariableOp?6feed_forward_sub_net_14/dense_72/MatMul/ReadVariableOp?6feed_forward_sub_net_14/dense_73/MatMul/ReadVariableOp?7feed_forward_sub_net_14/dense_74/BiasAdd/ReadVariableOp?6feed_forward_sub_net_14/dense_74/MatMul/ReadVariableOp?
Gfeed_forward_sub_net_14/batch_normalization_84/batchnorm/ReadVariableOpReadVariableOpPfeed_forward_sub_net_14_batch_normalization_84_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype02I
Gfeed_forward_sub_net_14/batch_normalization_84/batchnorm/ReadVariableOp?
>feed_forward_sub_net_14/batch_normalization_84/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2@
>feed_forward_sub_net_14/batch_normalization_84/batchnorm/add/y?
<feed_forward_sub_net_14/batch_normalization_84/batchnorm/addAddV2Ofeed_forward_sub_net_14/batch_normalization_84/batchnorm/ReadVariableOp:value:0Gfeed_forward_sub_net_14/batch_normalization_84/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2>
<feed_forward_sub_net_14/batch_normalization_84/batchnorm/add?
>feed_forward_sub_net_14/batch_normalization_84/batchnorm/RsqrtRsqrt@feed_forward_sub_net_14/batch_normalization_84/batchnorm/add:z:0*
T0*
_output_shapes
:
2@
>feed_forward_sub_net_14/batch_normalization_84/batchnorm/Rsqrt?
Kfeed_forward_sub_net_14/batch_normalization_84/batchnorm/mul/ReadVariableOpReadVariableOpTfeed_forward_sub_net_14_batch_normalization_84_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype02M
Kfeed_forward_sub_net_14/batch_normalization_84/batchnorm/mul/ReadVariableOp?
<feed_forward_sub_net_14/batch_normalization_84/batchnorm/mulMulBfeed_forward_sub_net_14/batch_normalization_84/batchnorm/Rsqrt:y:0Sfeed_forward_sub_net_14/batch_normalization_84/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2>
<feed_forward_sub_net_14/batch_normalization_84/batchnorm/mul?
>feed_forward_sub_net_14/batch_normalization_84/batchnorm/mul_1Mulinput_1@feed_forward_sub_net_14/batch_normalization_84/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????
2@
>feed_forward_sub_net_14/batch_normalization_84/batchnorm/mul_1?
Ifeed_forward_sub_net_14/batch_normalization_84/batchnorm/ReadVariableOp_1ReadVariableOpRfeed_forward_sub_net_14_batch_normalization_84_batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype02K
Ifeed_forward_sub_net_14/batch_normalization_84/batchnorm/ReadVariableOp_1?
>feed_forward_sub_net_14/batch_normalization_84/batchnorm/mul_2MulQfeed_forward_sub_net_14/batch_normalization_84/batchnorm/ReadVariableOp_1:value:0@feed_forward_sub_net_14/batch_normalization_84/batchnorm/mul:z:0*
T0*
_output_shapes
:
2@
>feed_forward_sub_net_14/batch_normalization_84/batchnorm/mul_2?
Ifeed_forward_sub_net_14/batch_normalization_84/batchnorm/ReadVariableOp_2ReadVariableOpRfeed_forward_sub_net_14_batch_normalization_84_batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype02K
Ifeed_forward_sub_net_14/batch_normalization_84/batchnorm/ReadVariableOp_2?
<feed_forward_sub_net_14/batch_normalization_84/batchnorm/subSubQfeed_forward_sub_net_14/batch_normalization_84/batchnorm/ReadVariableOp_2:value:0Bfeed_forward_sub_net_14/batch_normalization_84/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2>
<feed_forward_sub_net_14/batch_normalization_84/batchnorm/sub?
>feed_forward_sub_net_14/batch_normalization_84/batchnorm/add_1AddV2Bfeed_forward_sub_net_14/batch_normalization_84/batchnorm/mul_1:z:0@feed_forward_sub_net_14/batch_normalization_84/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????
2@
>feed_forward_sub_net_14/batch_normalization_84/batchnorm/add_1?
6feed_forward_sub_net_14/dense_70/MatMul/ReadVariableOpReadVariableOp?feed_forward_sub_net_14_dense_70_matmul_readvariableop_resource*
_output_shapes

:
*
dtype028
6feed_forward_sub_net_14/dense_70/MatMul/ReadVariableOp?
'feed_forward_sub_net_14/dense_70/MatMulMatMulBfeed_forward_sub_net_14/batch_normalization_84/batchnorm/add_1:z:0>feed_forward_sub_net_14/dense_70/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2)
'feed_forward_sub_net_14/dense_70/MatMul?
Gfeed_forward_sub_net_14/batch_normalization_85/batchnorm/ReadVariableOpReadVariableOpPfeed_forward_sub_net_14_batch_normalization_85_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02I
Gfeed_forward_sub_net_14/batch_normalization_85/batchnorm/ReadVariableOp?
>feed_forward_sub_net_14/batch_normalization_85/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2@
>feed_forward_sub_net_14/batch_normalization_85/batchnorm/add/y?
<feed_forward_sub_net_14/batch_normalization_85/batchnorm/addAddV2Ofeed_forward_sub_net_14/batch_normalization_85/batchnorm/ReadVariableOp:value:0Gfeed_forward_sub_net_14/batch_normalization_85/batchnorm/add/y:output:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_14/batch_normalization_85/batchnorm/add?
>feed_forward_sub_net_14/batch_normalization_85/batchnorm/RsqrtRsqrt@feed_forward_sub_net_14/batch_normalization_85/batchnorm/add:z:0*
T0*
_output_shapes
:2@
>feed_forward_sub_net_14/batch_normalization_85/batchnorm/Rsqrt?
Kfeed_forward_sub_net_14/batch_normalization_85/batchnorm/mul/ReadVariableOpReadVariableOpTfeed_forward_sub_net_14_batch_normalization_85_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02M
Kfeed_forward_sub_net_14/batch_normalization_85/batchnorm/mul/ReadVariableOp?
<feed_forward_sub_net_14/batch_normalization_85/batchnorm/mulMulBfeed_forward_sub_net_14/batch_normalization_85/batchnorm/Rsqrt:y:0Sfeed_forward_sub_net_14/batch_normalization_85/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_14/batch_normalization_85/batchnorm/mul?
>feed_forward_sub_net_14/batch_normalization_85/batchnorm/mul_1Mul1feed_forward_sub_net_14/dense_70/MatMul:product:0@feed_forward_sub_net_14/batch_normalization_85/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2@
>feed_forward_sub_net_14/batch_normalization_85/batchnorm/mul_1?
Ifeed_forward_sub_net_14/batch_normalization_85/batchnorm/ReadVariableOp_1ReadVariableOpRfeed_forward_sub_net_14_batch_normalization_85_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02K
Ifeed_forward_sub_net_14/batch_normalization_85/batchnorm/ReadVariableOp_1?
>feed_forward_sub_net_14/batch_normalization_85/batchnorm/mul_2MulQfeed_forward_sub_net_14/batch_normalization_85/batchnorm/ReadVariableOp_1:value:0@feed_forward_sub_net_14/batch_normalization_85/batchnorm/mul:z:0*
T0*
_output_shapes
:2@
>feed_forward_sub_net_14/batch_normalization_85/batchnorm/mul_2?
Ifeed_forward_sub_net_14/batch_normalization_85/batchnorm/ReadVariableOp_2ReadVariableOpRfeed_forward_sub_net_14_batch_normalization_85_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02K
Ifeed_forward_sub_net_14/batch_normalization_85/batchnorm/ReadVariableOp_2?
<feed_forward_sub_net_14/batch_normalization_85/batchnorm/subSubQfeed_forward_sub_net_14/batch_normalization_85/batchnorm/ReadVariableOp_2:value:0Bfeed_forward_sub_net_14/batch_normalization_85/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_14/batch_normalization_85/batchnorm/sub?
>feed_forward_sub_net_14/batch_normalization_85/batchnorm/add_1AddV2Bfeed_forward_sub_net_14/batch_normalization_85/batchnorm/mul_1:z:0@feed_forward_sub_net_14/batch_normalization_85/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2@
>feed_forward_sub_net_14/batch_normalization_85/batchnorm/add_1?
feed_forward_sub_net_14/ReluReluBfeed_forward_sub_net_14/batch_normalization_85/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
feed_forward_sub_net_14/Relu?
6feed_forward_sub_net_14/dense_71/MatMul/ReadVariableOpReadVariableOp?feed_forward_sub_net_14_dense_71_matmul_readvariableop_resource*
_output_shapes

:*
dtype028
6feed_forward_sub_net_14/dense_71/MatMul/ReadVariableOp?
'feed_forward_sub_net_14/dense_71/MatMulMatMul*feed_forward_sub_net_14/Relu:activations:0>feed_forward_sub_net_14/dense_71/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2)
'feed_forward_sub_net_14/dense_71/MatMul?
Gfeed_forward_sub_net_14/batch_normalization_86/batchnorm/ReadVariableOpReadVariableOpPfeed_forward_sub_net_14_batch_normalization_86_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02I
Gfeed_forward_sub_net_14/batch_normalization_86/batchnorm/ReadVariableOp?
>feed_forward_sub_net_14/batch_normalization_86/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2@
>feed_forward_sub_net_14/batch_normalization_86/batchnorm/add/y?
<feed_forward_sub_net_14/batch_normalization_86/batchnorm/addAddV2Ofeed_forward_sub_net_14/batch_normalization_86/batchnorm/ReadVariableOp:value:0Gfeed_forward_sub_net_14/batch_normalization_86/batchnorm/add/y:output:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_14/batch_normalization_86/batchnorm/add?
>feed_forward_sub_net_14/batch_normalization_86/batchnorm/RsqrtRsqrt@feed_forward_sub_net_14/batch_normalization_86/batchnorm/add:z:0*
T0*
_output_shapes
:2@
>feed_forward_sub_net_14/batch_normalization_86/batchnorm/Rsqrt?
Kfeed_forward_sub_net_14/batch_normalization_86/batchnorm/mul/ReadVariableOpReadVariableOpTfeed_forward_sub_net_14_batch_normalization_86_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02M
Kfeed_forward_sub_net_14/batch_normalization_86/batchnorm/mul/ReadVariableOp?
<feed_forward_sub_net_14/batch_normalization_86/batchnorm/mulMulBfeed_forward_sub_net_14/batch_normalization_86/batchnorm/Rsqrt:y:0Sfeed_forward_sub_net_14/batch_normalization_86/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_14/batch_normalization_86/batchnorm/mul?
>feed_forward_sub_net_14/batch_normalization_86/batchnorm/mul_1Mul1feed_forward_sub_net_14/dense_71/MatMul:product:0@feed_forward_sub_net_14/batch_normalization_86/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2@
>feed_forward_sub_net_14/batch_normalization_86/batchnorm/mul_1?
Ifeed_forward_sub_net_14/batch_normalization_86/batchnorm/ReadVariableOp_1ReadVariableOpRfeed_forward_sub_net_14_batch_normalization_86_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02K
Ifeed_forward_sub_net_14/batch_normalization_86/batchnorm/ReadVariableOp_1?
>feed_forward_sub_net_14/batch_normalization_86/batchnorm/mul_2MulQfeed_forward_sub_net_14/batch_normalization_86/batchnorm/ReadVariableOp_1:value:0@feed_forward_sub_net_14/batch_normalization_86/batchnorm/mul:z:0*
T0*
_output_shapes
:2@
>feed_forward_sub_net_14/batch_normalization_86/batchnorm/mul_2?
Ifeed_forward_sub_net_14/batch_normalization_86/batchnorm/ReadVariableOp_2ReadVariableOpRfeed_forward_sub_net_14_batch_normalization_86_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02K
Ifeed_forward_sub_net_14/batch_normalization_86/batchnorm/ReadVariableOp_2?
<feed_forward_sub_net_14/batch_normalization_86/batchnorm/subSubQfeed_forward_sub_net_14/batch_normalization_86/batchnorm/ReadVariableOp_2:value:0Bfeed_forward_sub_net_14/batch_normalization_86/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_14/batch_normalization_86/batchnorm/sub?
>feed_forward_sub_net_14/batch_normalization_86/batchnorm/add_1AddV2Bfeed_forward_sub_net_14/batch_normalization_86/batchnorm/mul_1:z:0@feed_forward_sub_net_14/batch_normalization_86/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2@
>feed_forward_sub_net_14/batch_normalization_86/batchnorm/add_1?
feed_forward_sub_net_14/Relu_1ReluBfeed_forward_sub_net_14/batch_normalization_86/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2 
feed_forward_sub_net_14/Relu_1?
6feed_forward_sub_net_14/dense_72/MatMul/ReadVariableOpReadVariableOp?feed_forward_sub_net_14_dense_72_matmul_readvariableop_resource*
_output_shapes

:*
dtype028
6feed_forward_sub_net_14/dense_72/MatMul/ReadVariableOp?
'feed_forward_sub_net_14/dense_72/MatMulMatMul,feed_forward_sub_net_14/Relu_1:activations:0>feed_forward_sub_net_14/dense_72/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2)
'feed_forward_sub_net_14/dense_72/MatMul?
Gfeed_forward_sub_net_14/batch_normalization_87/batchnorm/ReadVariableOpReadVariableOpPfeed_forward_sub_net_14_batch_normalization_87_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02I
Gfeed_forward_sub_net_14/batch_normalization_87/batchnorm/ReadVariableOp?
>feed_forward_sub_net_14/batch_normalization_87/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2@
>feed_forward_sub_net_14/batch_normalization_87/batchnorm/add/y?
<feed_forward_sub_net_14/batch_normalization_87/batchnorm/addAddV2Ofeed_forward_sub_net_14/batch_normalization_87/batchnorm/ReadVariableOp:value:0Gfeed_forward_sub_net_14/batch_normalization_87/batchnorm/add/y:output:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_14/batch_normalization_87/batchnorm/add?
>feed_forward_sub_net_14/batch_normalization_87/batchnorm/RsqrtRsqrt@feed_forward_sub_net_14/batch_normalization_87/batchnorm/add:z:0*
T0*
_output_shapes
:2@
>feed_forward_sub_net_14/batch_normalization_87/batchnorm/Rsqrt?
Kfeed_forward_sub_net_14/batch_normalization_87/batchnorm/mul/ReadVariableOpReadVariableOpTfeed_forward_sub_net_14_batch_normalization_87_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02M
Kfeed_forward_sub_net_14/batch_normalization_87/batchnorm/mul/ReadVariableOp?
<feed_forward_sub_net_14/batch_normalization_87/batchnorm/mulMulBfeed_forward_sub_net_14/batch_normalization_87/batchnorm/Rsqrt:y:0Sfeed_forward_sub_net_14/batch_normalization_87/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_14/batch_normalization_87/batchnorm/mul?
>feed_forward_sub_net_14/batch_normalization_87/batchnorm/mul_1Mul1feed_forward_sub_net_14/dense_72/MatMul:product:0@feed_forward_sub_net_14/batch_normalization_87/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2@
>feed_forward_sub_net_14/batch_normalization_87/batchnorm/mul_1?
Ifeed_forward_sub_net_14/batch_normalization_87/batchnorm/ReadVariableOp_1ReadVariableOpRfeed_forward_sub_net_14_batch_normalization_87_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02K
Ifeed_forward_sub_net_14/batch_normalization_87/batchnorm/ReadVariableOp_1?
>feed_forward_sub_net_14/batch_normalization_87/batchnorm/mul_2MulQfeed_forward_sub_net_14/batch_normalization_87/batchnorm/ReadVariableOp_1:value:0@feed_forward_sub_net_14/batch_normalization_87/batchnorm/mul:z:0*
T0*
_output_shapes
:2@
>feed_forward_sub_net_14/batch_normalization_87/batchnorm/mul_2?
Ifeed_forward_sub_net_14/batch_normalization_87/batchnorm/ReadVariableOp_2ReadVariableOpRfeed_forward_sub_net_14_batch_normalization_87_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02K
Ifeed_forward_sub_net_14/batch_normalization_87/batchnorm/ReadVariableOp_2?
<feed_forward_sub_net_14/batch_normalization_87/batchnorm/subSubQfeed_forward_sub_net_14/batch_normalization_87/batchnorm/ReadVariableOp_2:value:0Bfeed_forward_sub_net_14/batch_normalization_87/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_14/batch_normalization_87/batchnorm/sub?
>feed_forward_sub_net_14/batch_normalization_87/batchnorm/add_1AddV2Bfeed_forward_sub_net_14/batch_normalization_87/batchnorm/mul_1:z:0@feed_forward_sub_net_14/batch_normalization_87/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2@
>feed_forward_sub_net_14/batch_normalization_87/batchnorm/add_1?
feed_forward_sub_net_14/Relu_2ReluBfeed_forward_sub_net_14/batch_normalization_87/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2 
feed_forward_sub_net_14/Relu_2?
6feed_forward_sub_net_14/dense_73/MatMul/ReadVariableOpReadVariableOp?feed_forward_sub_net_14_dense_73_matmul_readvariableop_resource*
_output_shapes

:*
dtype028
6feed_forward_sub_net_14/dense_73/MatMul/ReadVariableOp?
'feed_forward_sub_net_14/dense_73/MatMulMatMul,feed_forward_sub_net_14/Relu_2:activations:0>feed_forward_sub_net_14/dense_73/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2)
'feed_forward_sub_net_14/dense_73/MatMul?
Gfeed_forward_sub_net_14/batch_normalization_88/batchnorm/ReadVariableOpReadVariableOpPfeed_forward_sub_net_14_batch_normalization_88_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02I
Gfeed_forward_sub_net_14/batch_normalization_88/batchnorm/ReadVariableOp?
>feed_forward_sub_net_14/batch_normalization_88/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2@
>feed_forward_sub_net_14/batch_normalization_88/batchnorm/add/y?
<feed_forward_sub_net_14/batch_normalization_88/batchnorm/addAddV2Ofeed_forward_sub_net_14/batch_normalization_88/batchnorm/ReadVariableOp:value:0Gfeed_forward_sub_net_14/batch_normalization_88/batchnorm/add/y:output:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_14/batch_normalization_88/batchnorm/add?
>feed_forward_sub_net_14/batch_normalization_88/batchnorm/RsqrtRsqrt@feed_forward_sub_net_14/batch_normalization_88/batchnorm/add:z:0*
T0*
_output_shapes
:2@
>feed_forward_sub_net_14/batch_normalization_88/batchnorm/Rsqrt?
Kfeed_forward_sub_net_14/batch_normalization_88/batchnorm/mul/ReadVariableOpReadVariableOpTfeed_forward_sub_net_14_batch_normalization_88_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02M
Kfeed_forward_sub_net_14/batch_normalization_88/batchnorm/mul/ReadVariableOp?
<feed_forward_sub_net_14/batch_normalization_88/batchnorm/mulMulBfeed_forward_sub_net_14/batch_normalization_88/batchnorm/Rsqrt:y:0Sfeed_forward_sub_net_14/batch_normalization_88/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_14/batch_normalization_88/batchnorm/mul?
>feed_forward_sub_net_14/batch_normalization_88/batchnorm/mul_1Mul1feed_forward_sub_net_14/dense_73/MatMul:product:0@feed_forward_sub_net_14/batch_normalization_88/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2@
>feed_forward_sub_net_14/batch_normalization_88/batchnorm/mul_1?
Ifeed_forward_sub_net_14/batch_normalization_88/batchnorm/ReadVariableOp_1ReadVariableOpRfeed_forward_sub_net_14_batch_normalization_88_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02K
Ifeed_forward_sub_net_14/batch_normalization_88/batchnorm/ReadVariableOp_1?
>feed_forward_sub_net_14/batch_normalization_88/batchnorm/mul_2MulQfeed_forward_sub_net_14/batch_normalization_88/batchnorm/ReadVariableOp_1:value:0@feed_forward_sub_net_14/batch_normalization_88/batchnorm/mul:z:0*
T0*
_output_shapes
:2@
>feed_forward_sub_net_14/batch_normalization_88/batchnorm/mul_2?
Ifeed_forward_sub_net_14/batch_normalization_88/batchnorm/ReadVariableOp_2ReadVariableOpRfeed_forward_sub_net_14_batch_normalization_88_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02K
Ifeed_forward_sub_net_14/batch_normalization_88/batchnorm/ReadVariableOp_2?
<feed_forward_sub_net_14/batch_normalization_88/batchnorm/subSubQfeed_forward_sub_net_14/batch_normalization_88/batchnorm/ReadVariableOp_2:value:0Bfeed_forward_sub_net_14/batch_normalization_88/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_14/batch_normalization_88/batchnorm/sub?
>feed_forward_sub_net_14/batch_normalization_88/batchnorm/add_1AddV2Bfeed_forward_sub_net_14/batch_normalization_88/batchnorm/mul_1:z:0@feed_forward_sub_net_14/batch_normalization_88/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2@
>feed_forward_sub_net_14/batch_normalization_88/batchnorm/add_1?
feed_forward_sub_net_14/Relu_3ReluBfeed_forward_sub_net_14/batch_normalization_88/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2 
feed_forward_sub_net_14/Relu_3?
6feed_forward_sub_net_14/dense_74/MatMul/ReadVariableOpReadVariableOp?feed_forward_sub_net_14_dense_74_matmul_readvariableop_resource*
_output_shapes

:
*
dtype028
6feed_forward_sub_net_14/dense_74/MatMul/ReadVariableOp?
'feed_forward_sub_net_14/dense_74/MatMulMatMul,feed_forward_sub_net_14/Relu_3:activations:0>feed_forward_sub_net_14/dense_74/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2)
'feed_forward_sub_net_14/dense_74/MatMul?
7feed_forward_sub_net_14/dense_74/BiasAdd/ReadVariableOpReadVariableOp@feed_forward_sub_net_14_dense_74_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype029
7feed_forward_sub_net_14/dense_74/BiasAdd/ReadVariableOp?
(feed_forward_sub_net_14/dense_74/BiasAddBiasAdd1feed_forward_sub_net_14/dense_74/MatMul:product:0?feed_forward_sub_net_14/dense_74/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2*
(feed_forward_sub_net_14/dense_74/BiasAdd?
IdentityIdentity1feed_forward_sub_net_14/dense_74/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOpH^feed_forward_sub_net_14/batch_normalization_84/batchnorm/ReadVariableOpJ^feed_forward_sub_net_14/batch_normalization_84/batchnorm/ReadVariableOp_1J^feed_forward_sub_net_14/batch_normalization_84/batchnorm/ReadVariableOp_2L^feed_forward_sub_net_14/batch_normalization_84/batchnorm/mul/ReadVariableOpH^feed_forward_sub_net_14/batch_normalization_85/batchnorm/ReadVariableOpJ^feed_forward_sub_net_14/batch_normalization_85/batchnorm/ReadVariableOp_1J^feed_forward_sub_net_14/batch_normalization_85/batchnorm/ReadVariableOp_2L^feed_forward_sub_net_14/batch_normalization_85/batchnorm/mul/ReadVariableOpH^feed_forward_sub_net_14/batch_normalization_86/batchnorm/ReadVariableOpJ^feed_forward_sub_net_14/batch_normalization_86/batchnorm/ReadVariableOp_1J^feed_forward_sub_net_14/batch_normalization_86/batchnorm/ReadVariableOp_2L^feed_forward_sub_net_14/batch_normalization_86/batchnorm/mul/ReadVariableOpH^feed_forward_sub_net_14/batch_normalization_87/batchnorm/ReadVariableOpJ^feed_forward_sub_net_14/batch_normalization_87/batchnorm/ReadVariableOp_1J^feed_forward_sub_net_14/batch_normalization_87/batchnorm/ReadVariableOp_2L^feed_forward_sub_net_14/batch_normalization_87/batchnorm/mul/ReadVariableOpH^feed_forward_sub_net_14/batch_normalization_88/batchnorm/ReadVariableOpJ^feed_forward_sub_net_14/batch_normalization_88/batchnorm/ReadVariableOp_1J^feed_forward_sub_net_14/batch_normalization_88/batchnorm/ReadVariableOp_2L^feed_forward_sub_net_14/batch_normalization_88/batchnorm/mul/ReadVariableOp7^feed_forward_sub_net_14/dense_70/MatMul/ReadVariableOp7^feed_forward_sub_net_14/dense_71/MatMul/ReadVariableOp7^feed_forward_sub_net_14/dense_72/MatMul/ReadVariableOp7^feed_forward_sub_net_14/dense_73/MatMul/ReadVariableOp8^feed_forward_sub_net_14/dense_74/BiasAdd/ReadVariableOp7^feed_forward_sub_net_14/dense_74/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????
: : : : : : : : : : : : : : : : : : : : : : : : : : 2?
Gfeed_forward_sub_net_14/batch_normalization_84/batchnorm/ReadVariableOpGfeed_forward_sub_net_14/batch_normalization_84/batchnorm/ReadVariableOp2?
Ifeed_forward_sub_net_14/batch_normalization_84/batchnorm/ReadVariableOp_1Ifeed_forward_sub_net_14/batch_normalization_84/batchnorm/ReadVariableOp_12?
Ifeed_forward_sub_net_14/batch_normalization_84/batchnorm/ReadVariableOp_2Ifeed_forward_sub_net_14/batch_normalization_84/batchnorm/ReadVariableOp_22?
Kfeed_forward_sub_net_14/batch_normalization_84/batchnorm/mul/ReadVariableOpKfeed_forward_sub_net_14/batch_normalization_84/batchnorm/mul/ReadVariableOp2?
Gfeed_forward_sub_net_14/batch_normalization_85/batchnorm/ReadVariableOpGfeed_forward_sub_net_14/batch_normalization_85/batchnorm/ReadVariableOp2?
Ifeed_forward_sub_net_14/batch_normalization_85/batchnorm/ReadVariableOp_1Ifeed_forward_sub_net_14/batch_normalization_85/batchnorm/ReadVariableOp_12?
Ifeed_forward_sub_net_14/batch_normalization_85/batchnorm/ReadVariableOp_2Ifeed_forward_sub_net_14/batch_normalization_85/batchnorm/ReadVariableOp_22?
Kfeed_forward_sub_net_14/batch_normalization_85/batchnorm/mul/ReadVariableOpKfeed_forward_sub_net_14/batch_normalization_85/batchnorm/mul/ReadVariableOp2?
Gfeed_forward_sub_net_14/batch_normalization_86/batchnorm/ReadVariableOpGfeed_forward_sub_net_14/batch_normalization_86/batchnorm/ReadVariableOp2?
Ifeed_forward_sub_net_14/batch_normalization_86/batchnorm/ReadVariableOp_1Ifeed_forward_sub_net_14/batch_normalization_86/batchnorm/ReadVariableOp_12?
Ifeed_forward_sub_net_14/batch_normalization_86/batchnorm/ReadVariableOp_2Ifeed_forward_sub_net_14/batch_normalization_86/batchnorm/ReadVariableOp_22?
Kfeed_forward_sub_net_14/batch_normalization_86/batchnorm/mul/ReadVariableOpKfeed_forward_sub_net_14/batch_normalization_86/batchnorm/mul/ReadVariableOp2?
Gfeed_forward_sub_net_14/batch_normalization_87/batchnorm/ReadVariableOpGfeed_forward_sub_net_14/batch_normalization_87/batchnorm/ReadVariableOp2?
Ifeed_forward_sub_net_14/batch_normalization_87/batchnorm/ReadVariableOp_1Ifeed_forward_sub_net_14/batch_normalization_87/batchnorm/ReadVariableOp_12?
Ifeed_forward_sub_net_14/batch_normalization_87/batchnorm/ReadVariableOp_2Ifeed_forward_sub_net_14/batch_normalization_87/batchnorm/ReadVariableOp_22?
Kfeed_forward_sub_net_14/batch_normalization_87/batchnorm/mul/ReadVariableOpKfeed_forward_sub_net_14/batch_normalization_87/batchnorm/mul/ReadVariableOp2?
Gfeed_forward_sub_net_14/batch_normalization_88/batchnorm/ReadVariableOpGfeed_forward_sub_net_14/batch_normalization_88/batchnorm/ReadVariableOp2?
Ifeed_forward_sub_net_14/batch_normalization_88/batchnorm/ReadVariableOp_1Ifeed_forward_sub_net_14/batch_normalization_88/batchnorm/ReadVariableOp_12?
Ifeed_forward_sub_net_14/batch_normalization_88/batchnorm/ReadVariableOp_2Ifeed_forward_sub_net_14/batch_normalization_88/batchnorm/ReadVariableOp_22?
Kfeed_forward_sub_net_14/batch_normalization_88/batchnorm/mul/ReadVariableOpKfeed_forward_sub_net_14/batch_normalization_88/batchnorm/mul/ReadVariableOp2p
6feed_forward_sub_net_14/dense_70/MatMul/ReadVariableOp6feed_forward_sub_net_14/dense_70/MatMul/ReadVariableOp2p
6feed_forward_sub_net_14/dense_71/MatMul/ReadVariableOp6feed_forward_sub_net_14/dense_71/MatMul/ReadVariableOp2p
6feed_forward_sub_net_14/dense_72/MatMul/ReadVariableOp6feed_forward_sub_net_14/dense_72/MatMul/ReadVariableOp2p
6feed_forward_sub_net_14/dense_73/MatMul/ReadVariableOp6feed_forward_sub_net_14/dense_73/MatMul/ReadVariableOp2r
7feed_forward_sub_net_14/dense_74/BiasAdd/ReadVariableOp7feed_forward_sub_net_14/dense_74/BiasAdd/ReadVariableOp2p
6feed_forward_sub_net_14/dense_74/MatMul/ReadVariableOp6feed_forward_sub_net_14/dense_74/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????

!
_user_specified_name	input_1
?
~
*__inference_dense_70_layer_call_fn_7063167

inputs
unknown:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_70_layer_call_and_return_conditional_losses_70613072
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????
: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?E
?
T__inference_feed_forward_sub_net_14_layer_call_and_return_conditional_losses_7061401
x,
batch_normalization_84_7061291:
,
batch_normalization_84_7061293:
,
batch_normalization_84_7061295:
,
batch_normalization_84_7061297:
"
dense_70_7061308:
,
batch_normalization_85_7061311:,
batch_normalization_85_7061313:,
batch_normalization_85_7061315:,
batch_normalization_85_7061317:"
dense_71_7061329:,
batch_normalization_86_7061332:,
batch_normalization_86_7061334:,
batch_normalization_86_7061336:,
batch_normalization_86_7061338:"
dense_72_7061350:,
batch_normalization_87_7061353:,
batch_normalization_87_7061355:,
batch_normalization_87_7061357:,
batch_normalization_87_7061359:"
dense_73_7061371:,
batch_normalization_88_7061374:,
batch_normalization_88_7061376:,
batch_normalization_88_7061378:,
batch_normalization_88_7061380:"
dense_74_7061395:

dense_74_7061397:

identity??.batch_normalization_84/StatefulPartitionedCall?.batch_normalization_85/StatefulPartitionedCall?.batch_normalization_86/StatefulPartitionedCall?.batch_normalization_87/StatefulPartitionedCall?.batch_normalization_88/StatefulPartitionedCall? dense_70/StatefulPartitionedCall? dense_71/StatefulPartitionedCall? dense_72/StatefulPartitionedCall? dense_73/StatefulPartitionedCall? dense_74/StatefulPartitionedCall?
.batch_normalization_84/StatefulPartitionedCallStatefulPartitionedCallxbatch_normalization_84_7061291batch_normalization_84_7061293batch_normalization_84_7061295batch_normalization_84_7061297*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_84_layer_call_and_return_conditional_losses_706047820
.batch_normalization_84/StatefulPartitionedCall?
 dense_70/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_84/StatefulPartitionedCall:output:0dense_70_7061308*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_70_layer_call_and_return_conditional_losses_70613072"
 dense_70/StatefulPartitionedCall?
.batch_normalization_85/StatefulPartitionedCallStatefulPartitionedCall)dense_70/StatefulPartitionedCall:output:0batch_normalization_85_7061311batch_normalization_85_7061313batch_normalization_85_7061315batch_normalization_85_7061317*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_85_layer_call_and_return_conditional_losses_706064420
.batch_normalization_85/StatefulPartitionedCall
ReluRelu7batch_normalization_85/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
Relu?
 dense_71/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0dense_71_7061329*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_71_layer_call_and_return_conditional_losses_70613282"
 dense_71/StatefulPartitionedCall?
.batch_normalization_86/StatefulPartitionedCallStatefulPartitionedCall)dense_71/StatefulPartitionedCall:output:0batch_normalization_86_7061332batch_normalization_86_7061334batch_normalization_86_7061336batch_normalization_86_7061338*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_86_layer_call_and_return_conditional_losses_706081020
.batch_normalization_86/StatefulPartitionedCall?
Relu_1Relu7batch_normalization_86/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
Relu_1?
 dense_72/StatefulPartitionedCallStatefulPartitionedCallRelu_1:activations:0dense_72_7061350*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_72_layer_call_and_return_conditional_losses_70613492"
 dense_72/StatefulPartitionedCall?
.batch_normalization_87/StatefulPartitionedCallStatefulPartitionedCall)dense_72/StatefulPartitionedCall:output:0batch_normalization_87_7061353batch_normalization_87_7061355batch_normalization_87_7061357batch_normalization_87_7061359*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_87_layer_call_and_return_conditional_losses_706097620
.batch_normalization_87/StatefulPartitionedCall?
Relu_2Relu7batch_normalization_87/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
Relu_2?
 dense_73/StatefulPartitionedCallStatefulPartitionedCallRelu_2:activations:0dense_73_7061371*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_73_layer_call_and_return_conditional_losses_70613702"
 dense_73/StatefulPartitionedCall?
.batch_normalization_88/StatefulPartitionedCallStatefulPartitionedCall)dense_73/StatefulPartitionedCall:output:0batch_normalization_88_7061374batch_normalization_88_7061376batch_normalization_88_7061378batch_normalization_88_7061380*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_88_layer_call_and_return_conditional_losses_706114220
.batch_normalization_88/StatefulPartitionedCall?
Relu_3Relu7batch_normalization_88/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
Relu_3?
 dense_74/StatefulPartitionedCallStatefulPartitionedCallRelu_3:activations:0dense_74_7061395dense_74_7061397*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_74_layer_call_and_return_conditional_losses_70613942"
 dense_74/StatefulPartitionedCall?
IdentityIdentity)dense_74/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp/^batch_normalization_84/StatefulPartitionedCall/^batch_normalization_85/StatefulPartitionedCall/^batch_normalization_86/StatefulPartitionedCall/^batch_normalization_87/StatefulPartitionedCall/^batch_normalization_88/StatefulPartitionedCall!^dense_70/StatefulPartitionedCall!^dense_71/StatefulPartitionedCall!^dense_72/StatefulPartitionedCall!^dense_73/StatefulPartitionedCall!^dense_74/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????
: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_84/StatefulPartitionedCall.batch_normalization_84/StatefulPartitionedCall2`
.batch_normalization_85/StatefulPartitionedCall.batch_normalization_85/StatefulPartitionedCall2`
.batch_normalization_86/StatefulPartitionedCall.batch_normalization_86/StatefulPartitionedCall2`
.batch_normalization_87/StatefulPartitionedCall.batch_normalization_87/StatefulPartitionedCall2`
.batch_normalization_88/StatefulPartitionedCall.batch_normalization_88/StatefulPartitionedCall2D
 dense_70/StatefulPartitionedCall dense_70/StatefulPartitionedCall2D
 dense_71/StatefulPartitionedCall dense_71/StatefulPartitionedCall2D
 dense_72/StatefulPartitionedCall dense_72/StatefulPartitionedCall2D
 dense_73/StatefulPartitionedCall dense_73/StatefulPartitionedCall2D
 dense_74/StatefulPartitionedCall dense_74/StatefulPartitionedCall:J F
'
_output_shapes
:?????????


_user_specified_namex
??
?
T__inference_feed_forward_sub_net_14_layer_call_and_return_conditional_losses_7062750
input_1L
>batch_normalization_84_assignmovingavg_readvariableop_resource:
N
@batch_normalization_84_assignmovingavg_1_readvariableop_resource:
J
<batch_normalization_84_batchnorm_mul_readvariableop_resource:
F
8batch_normalization_84_batchnorm_readvariableop_resource:
9
'dense_70_matmul_readvariableop_resource:
L
>batch_normalization_85_assignmovingavg_readvariableop_resource:N
@batch_normalization_85_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_85_batchnorm_mul_readvariableop_resource:F
8batch_normalization_85_batchnorm_readvariableop_resource:9
'dense_71_matmul_readvariableop_resource:L
>batch_normalization_86_assignmovingavg_readvariableop_resource:N
@batch_normalization_86_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_86_batchnorm_mul_readvariableop_resource:F
8batch_normalization_86_batchnorm_readvariableop_resource:9
'dense_72_matmul_readvariableop_resource:L
>batch_normalization_87_assignmovingavg_readvariableop_resource:N
@batch_normalization_87_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_87_batchnorm_mul_readvariableop_resource:F
8batch_normalization_87_batchnorm_readvariableop_resource:9
'dense_73_matmul_readvariableop_resource:L
>batch_normalization_88_assignmovingavg_readvariableop_resource:N
@batch_normalization_88_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_88_batchnorm_mul_readvariableop_resource:F
8batch_normalization_88_batchnorm_readvariableop_resource:9
'dense_74_matmul_readvariableop_resource:
6
(dense_74_biasadd_readvariableop_resource:

identity??&batch_normalization_84/AssignMovingAvg?5batch_normalization_84/AssignMovingAvg/ReadVariableOp?(batch_normalization_84/AssignMovingAvg_1?7batch_normalization_84/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_84/batchnorm/ReadVariableOp?3batch_normalization_84/batchnorm/mul/ReadVariableOp?&batch_normalization_85/AssignMovingAvg?5batch_normalization_85/AssignMovingAvg/ReadVariableOp?(batch_normalization_85/AssignMovingAvg_1?7batch_normalization_85/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_85/batchnorm/ReadVariableOp?3batch_normalization_85/batchnorm/mul/ReadVariableOp?&batch_normalization_86/AssignMovingAvg?5batch_normalization_86/AssignMovingAvg/ReadVariableOp?(batch_normalization_86/AssignMovingAvg_1?7batch_normalization_86/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_86/batchnorm/ReadVariableOp?3batch_normalization_86/batchnorm/mul/ReadVariableOp?&batch_normalization_87/AssignMovingAvg?5batch_normalization_87/AssignMovingAvg/ReadVariableOp?(batch_normalization_87/AssignMovingAvg_1?7batch_normalization_87/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_87/batchnorm/ReadVariableOp?3batch_normalization_87/batchnorm/mul/ReadVariableOp?&batch_normalization_88/AssignMovingAvg?5batch_normalization_88/AssignMovingAvg/ReadVariableOp?(batch_normalization_88/AssignMovingAvg_1?7batch_normalization_88/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_88/batchnorm/ReadVariableOp?3batch_normalization_88/batchnorm/mul/ReadVariableOp?dense_70/MatMul/ReadVariableOp?dense_71/MatMul/ReadVariableOp?dense_72/MatMul/ReadVariableOp?dense_73/MatMul/ReadVariableOp?dense_74/BiasAdd/ReadVariableOp?dense_74/MatMul/ReadVariableOp?
5batch_normalization_84/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_84/moments/mean/reduction_indices?
#batch_normalization_84/moments/meanMeaninput_1>batch_normalization_84/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2%
#batch_normalization_84/moments/mean?
+batch_normalization_84/moments/StopGradientStopGradient,batch_normalization_84/moments/mean:output:0*
T0*
_output_shapes

:
2-
+batch_normalization_84/moments/StopGradient?
0batch_normalization_84/moments/SquaredDifferenceSquaredDifferenceinput_14batch_normalization_84/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????
22
0batch_normalization_84/moments/SquaredDifference?
9batch_normalization_84/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_84/moments/variance/reduction_indices?
'batch_normalization_84/moments/varianceMean4batch_normalization_84/moments/SquaredDifference:z:0Bbatch_normalization_84/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2)
'batch_normalization_84/moments/variance?
&batch_normalization_84/moments/SqueezeSqueeze,batch_normalization_84/moments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2(
&batch_normalization_84/moments/Squeeze?
(batch_normalization_84/moments/Squeeze_1Squeeze0batch_normalization_84/moments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2*
(batch_normalization_84/moments/Squeeze_1?
,batch_normalization_84/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_84/AssignMovingAvg/decay?
+batch_normalization_84/AssignMovingAvg/CastCast5batch_normalization_84/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_84/AssignMovingAvg/Cast?
5batch_normalization_84/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_84_assignmovingavg_readvariableop_resource*
_output_shapes
:
*
dtype027
5batch_normalization_84/AssignMovingAvg/ReadVariableOp?
*batch_normalization_84/AssignMovingAvg/subSub=batch_normalization_84/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_84/moments/Squeeze:output:0*
T0*
_output_shapes
:
2,
*batch_normalization_84/AssignMovingAvg/sub?
*batch_normalization_84/AssignMovingAvg/mulMul.batch_normalization_84/AssignMovingAvg/sub:z:0/batch_normalization_84/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:
2,
*batch_normalization_84/AssignMovingAvg/mul?
&batch_normalization_84/AssignMovingAvgAssignSubVariableOp>batch_normalization_84_assignmovingavg_readvariableop_resource.batch_normalization_84/AssignMovingAvg/mul:z:06^batch_normalization_84/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_84/AssignMovingAvg?
.batch_normalization_84/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_84/AssignMovingAvg_1/decay?
-batch_normalization_84/AssignMovingAvg_1/CastCast7batch_normalization_84/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_84/AssignMovingAvg_1/Cast?
7batch_normalization_84/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_84_assignmovingavg_1_readvariableop_resource*
_output_shapes
:
*
dtype029
7batch_normalization_84/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_84/AssignMovingAvg_1/subSub?batch_normalization_84/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_84/moments/Squeeze_1:output:0*
T0*
_output_shapes
:
2.
,batch_normalization_84/AssignMovingAvg_1/sub?
,batch_normalization_84/AssignMovingAvg_1/mulMul0batch_normalization_84/AssignMovingAvg_1/sub:z:01batch_normalization_84/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:
2.
,batch_normalization_84/AssignMovingAvg_1/mul?
(batch_normalization_84/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_84_assignmovingavg_1_readvariableop_resource0batch_normalization_84/AssignMovingAvg_1/mul:z:08^batch_normalization_84/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_84/AssignMovingAvg_1?
&batch_normalization_84/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2(
&batch_normalization_84/batchnorm/add/y?
$batch_normalization_84/batchnorm/addAddV21batch_normalization_84/moments/Squeeze_1:output:0/batch_normalization_84/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2&
$batch_normalization_84/batchnorm/add?
&batch_normalization_84/batchnorm/RsqrtRsqrt(batch_normalization_84/batchnorm/add:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_84/batchnorm/Rsqrt?
3batch_normalization_84/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_84_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype025
3batch_normalization_84/batchnorm/mul/ReadVariableOp?
$batch_normalization_84/batchnorm/mulMul*batch_normalization_84/batchnorm/Rsqrt:y:0;batch_normalization_84/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2&
$batch_normalization_84/batchnorm/mul?
&batch_normalization_84/batchnorm/mul_1Mulinput_1(batch_normalization_84/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????
2(
&batch_normalization_84/batchnorm/mul_1?
&batch_normalization_84/batchnorm/mul_2Mul/batch_normalization_84/moments/Squeeze:output:0(batch_normalization_84/batchnorm/mul:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_84/batchnorm/mul_2?
/batch_normalization_84/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_84_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype021
/batch_normalization_84/batchnorm/ReadVariableOp?
$batch_normalization_84/batchnorm/subSub7batch_normalization_84/batchnorm/ReadVariableOp:value:0*batch_normalization_84/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2&
$batch_normalization_84/batchnorm/sub?
&batch_normalization_84/batchnorm/add_1AddV2*batch_normalization_84/batchnorm/mul_1:z:0(batch_normalization_84/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????
2(
&batch_normalization_84/batchnorm/add_1?
dense_70/MatMul/ReadVariableOpReadVariableOp'dense_70_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_70/MatMul/ReadVariableOp?
dense_70/MatMulMatMul*batch_normalization_84/batchnorm/add_1:z:0&dense_70/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_70/MatMul?
5batch_normalization_85/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_85/moments/mean/reduction_indices?
#batch_normalization_85/moments/meanMeandense_70/MatMul:product:0>batch_normalization_85/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_85/moments/mean?
+batch_normalization_85/moments/StopGradientStopGradient,batch_normalization_85/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_85/moments/StopGradient?
0batch_normalization_85/moments/SquaredDifferenceSquaredDifferencedense_70/MatMul:product:04batch_normalization_85/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????22
0batch_normalization_85/moments/SquaredDifference?
9batch_normalization_85/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_85/moments/variance/reduction_indices?
'batch_normalization_85/moments/varianceMean4batch_normalization_85/moments/SquaredDifference:z:0Bbatch_normalization_85/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_85/moments/variance?
&batch_normalization_85/moments/SqueezeSqueeze,batch_normalization_85/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_85/moments/Squeeze?
(batch_normalization_85/moments/Squeeze_1Squeeze0batch_normalization_85/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_85/moments/Squeeze_1?
,batch_normalization_85/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_85/AssignMovingAvg/decay?
+batch_normalization_85/AssignMovingAvg/CastCast5batch_normalization_85/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_85/AssignMovingAvg/Cast?
5batch_normalization_85/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_85_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_85/AssignMovingAvg/ReadVariableOp?
*batch_normalization_85/AssignMovingAvg/subSub=batch_normalization_85/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_85/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_85/AssignMovingAvg/sub?
*batch_normalization_85/AssignMovingAvg/mulMul.batch_normalization_85/AssignMovingAvg/sub:z:0/batch_normalization_85/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_85/AssignMovingAvg/mul?
&batch_normalization_85/AssignMovingAvgAssignSubVariableOp>batch_normalization_85_assignmovingavg_readvariableop_resource.batch_normalization_85/AssignMovingAvg/mul:z:06^batch_normalization_85/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_85/AssignMovingAvg?
.batch_normalization_85/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_85/AssignMovingAvg_1/decay?
-batch_normalization_85/AssignMovingAvg_1/CastCast7batch_normalization_85/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_85/AssignMovingAvg_1/Cast?
7batch_normalization_85/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_85_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_85/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_85/AssignMovingAvg_1/subSub?batch_normalization_85/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_85/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_85/AssignMovingAvg_1/sub?
,batch_normalization_85/AssignMovingAvg_1/mulMul0batch_normalization_85/AssignMovingAvg_1/sub:z:01batch_normalization_85/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_85/AssignMovingAvg_1/mul?
(batch_normalization_85/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_85_assignmovingavg_1_readvariableop_resource0batch_normalization_85/AssignMovingAvg_1/mul:z:08^batch_normalization_85/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_85/AssignMovingAvg_1?
&batch_normalization_85/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2(
&batch_normalization_85/batchnorm/add/y?
$batch_normalization_85/batchnorm/addAddV21batch_normalization_85/moments/Squeeze_1:output:0/batch_normalization_85/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_85/batchnorm/add?
&batch_normalization_85/batchnorm/RsqrtRsqrt(batch_normalization_85/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_85/batchnorm/Rsqrt?
3batch_normalization_85/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_85_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_85/batchnorm/mul/ReadVariableOp?
$batch_normalization_85/batchnorm/mulMul*batch_normalization_85/batchnorm/Rsqrt:y:0;batch_normalization_85/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_85/batchnorm/mul?
&batch_normalization_85/batchnorm/mul_1Muldense_70/MatMul:product:0(batch_normalization_85/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_85/batchnorm/mul_1?
&batch_normalization_85/batchnorm/mul_2Mul/batch_normalization_85/moments/Squeeze:output:0(batch_normalization_85/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_85/batchnorm/mul_2?
/batch_normalization_85/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_85_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_85/batchnorm/ReadVariableOp?
$batch_normalization_85/batchnorm/subSub7batch_normalization_85/batchnorm/ReadVariableOp:value:0*batch_normalization_85/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_85/batchnorm/sub?
&batch_normalization_85/batchnorm/add_1AddV2*batch_normalization_85/batchnorm/mul_1:z:0(batch_normalization_85/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_85/batchnorm/add_1r
ReluRelu*batch_normalization_85/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu?
dense_71/MatMul/ReadVariableOpReadVariableOp'dense_71_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_71/MatMul/ReadVariableOp?
dense_71/MatMulMatMulRelu:activations:0&dense_71/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_71/MatMul?
5batch_normalization_86/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_86/moments/mean/reduction_indices?
#batch_normalization_86/moments/meanMeandense_71/MatMul:product:0>batch_normalization_86/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_86/moments/mean?
+batch_normalization_86/moments/StopGradientStopGradient,batch_normalization_86/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_86/moments/StopGradient?
0batch_normalization_86/moments/SquaredDifferenceSquaredDifferencedense_71/MatMul:product:04batch_normalization_86/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????22
0batch_normalization_86/moments/SquaredDifference?
9batch_normalization_86/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_86/moments/variance/reduction_indices?
'batch_normalization_86/moments/varianceMean4batch_normalization_86/moments/SquaredDifference:z:0Bbatch_normalization_86/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_86/moments/variance?
&batch_normalization_86/moments/SqueezeSqueeze,batch_normalization_86/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_86/moments/Squeeze?
(batch_normalization_86/moments/Squeeze_1Squeeze0batch_normalization_86/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_86/moments/Squeeze_1?
,batch_normalization_86/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_86/AssignMovingAvg/decay?
+batch_normalization_86/AssignMovingAvg/CastCast5batch_normalization_86/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_86/AssignMovingAvg/Cast?
5batch_normalization_86/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_86_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_86/AssignMovingAvg/ReadVariableOp?
*batch_normalization_86/AssignMovingAvg/subSub=batch_normalization_86/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_86/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_86/AssignMovingAvg/sub?
*batch_normalization_86/AssignMovingAvg/mulMul.batch_normalization_86/AssignMovingAvg/sub:z:0/batch_normalization_86/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_86/AssignMovingAvg/mul?
&batch_normalization_86/AssignMovingAvgAssignSubVariableOp>batch_normalization_86_assignmovingavg_readvariableop_resource.batch_normalization_86/AssignMovingAvg/mul:z:06^batch_normalization_86/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_86/AssignMovingAvg?
.batch_normalization_86/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_86/AssignMovingAvg_1/decay?
-batch_normalization_86/AssignMovingAvg_1/CastCast7batch_normalization_86/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_86/AssignMovingAvg_1/Cast?
7batch_normalization_86/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_86_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_86/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_86/AssignMovingAvg_1/subSub?batch_normalization_86/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_86/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_86/AssignMovingAvg_1/sub?
,batch_normalization_86/AssignMovingAvg_1/mulMul0batch_normalization_86/AssignMovingAvg_1/sub:z:01batch_normalization_86/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_86/AssignMovingAvg_1/mul?
(batch_normalization_86/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_86_assignmovingavg_1_readvariableop_resource0batch_normalization_86/AssignMovingAvg_1/mul:z:08^batch_normalization_86/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_86/AssignMovingAvg_1?
&batch_normalization_86/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2(
&batch_normalization_86/batchnorm/add/y?
$batch_normalization_86/batchnorm/addAddV21batch_normalization_86/moments/Squeeze_1:output:0/batch_normalization_86/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_86/batchnorm/add?
&batch_normalization_86/batchnorm/RsqrtRsqrt(batch_normalization_86/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_86/batchnorm/Rsqrt?
3batch_normalization_86/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_86_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_86/batchnorm/mul/ReadVariableOp?
$batch_normalization_86/batchnorm/mulMul*batch_normalization_86/batchnorm/Rsqrt:y:0;batch_normalization_86/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_86/batchnorm/mul?
&batch_normalization_86/batchnorm/mul_1Muldense_71/MatMul:product:0(batch_normalization_86/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_86/batchnorm/mul_1?
&batch_normalization_86/batchnorm/mul_2Mul/batch_normalization_86/moments/Squeeze:output:0(batch_normalization_86/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_86/batchnorm/mul_2?
/batch_normalization_86/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_86_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_86/batchnorm/ReadVariableOp?
$batch_normalization_86/batchnorm/subSub7batch_normalization_86/batchnorm/ReadVariableOp:value:0*batch_normalization_86/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_86/batchnorm/sub?
&batch_normalization_86/batchnorm/add_1AddV2*batch_normalization_86/batchnorm/mul_1:z:0(batch_normalization_86/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_86/batchnorm/add_1v
Relu_1Relu*batch_normalization_86/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu_1?
dense_72/MatMul/ReadVariableOpReadVariableOp'dense_72_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_72/MatMul/ReadVariableOp?
dense_72/MatMulMatMulRelu_1:activations:0&dense_72/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_72/MatMul?
5batch_normalization_87/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_87/moments/mean/reduction_indices?
#batch_normalization_87/moments/meanMeandense_72/MatMul:product:0>batch_normalization_87/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_87/moments/mean?
+batch_normalization_87/moments/StopGradientStopGradient,batch_normalization_87/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_87/moments/StopGradient?
0batch_normalization_87/moments/SquaredDifferenceSquaredDifferencedense_72/MatMul:product:04batch_normalization_87/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????22
0batch_normalization_87/moments/SquaredDifference?
9batch_normalization_87/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_87/moments/variance/reduction_indices?
'batch_normalization_87/moments/varianceMean4batch_normalization_87/moments/SquaredDifference:z:0Bbatch_normalization_87/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_87/moments/variance?
&batch_normalization_87/moments/SqueezeSqueeze,batch_normalization_87/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_87/moments/Squeeze?
(batch_normalization_87/moments/Squeeze_1Squeeze0batch_normalization_87/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_87/moments/Squeeze_1?
,batch_normalization_87/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_87/AssignMovingAvg/decay?
+batch_normalization_87/AssignMovingAvg/CastCast5batch_normalization_87/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_87/AssignMovingAvg/Cast?
5batch_normalization_87/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_87_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_87/AssignMovingAvg/ReadVariableOp?
*batch_normalization_87/AssignMovingAvg/subSub=batch_normalization_87/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_87/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_87/AssignMovingAvg/sub?
*batch_normalization_87/AssignMovingAvg/mulMul.batch_normalization_87/AssignMovingAvg/sub:z:0/batch_normalization_87/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_87/AssignMovingAvg/mul?
&batch_normalization_87/AssignMovingAvgAssignSubVariableOp>batch_normalization_87_assignmovingavg_readvariableop_resource.batch_normalization_87/AssignMovingAvg/mul:z:06^batch_normalization_87/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_87/AssignMovingAvg?
.batch_normalization_87/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_87/AssignMovingAvg_1/decay?
-batch_normalization_87/AssignMovingAvg_1/CastCast7batch_normalization_87/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_87/AssignMovingAvg_1/Cast?
7batch_normalization_87/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_87_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_87/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_87/AssignMovingAvg_1/subSub?batch_normalization_87/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_87/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_87/AssignMovingAvg_1/sub?
,batch_normalization_87/AssignMovingAvg_1/mulMul0batch_normalization_87/AssignMovingAvg_1/sub:z:01batch_normalization_87/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_87/AssignMovingAvg_1/mul?
(batch_normalization_87/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_87_assignmovingavg_1_readvariableop_resource0batch_normalization_87/AssignMovingAvg_1/mul:z:08^batch_normalization_87/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_87/AssignMovingAvg_1?
&batch_normalization_87/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2(
&batch_normalization_87/batchnorm/add/y?
$batch_normalization_87/batchnorm/addAddV21batch_normalization_87/moments/Squeeze_1:output:0/batch_normalization_87/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_87/batchnorm/add?
&batch_normalization_87/batchnorm/RsqrtRsqrt(batch_normalization_87/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_87/batchnorm/Rsqrt?
3batch_normalization_87/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_87_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_87/batchnorm/mul/ReadVariableOp?
$batch_normalization_87/batchnorm/mulMul*batch_normalization_87/batchnorm/Rsqrt:y:0;batch_normalization_87/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_87/batchnorm/mul?
&batch_normalization_87/batchnorm/mul_1Muldense_72/MatMul:product:0(batch_normalization_87/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_87/batchnorm/mul_1?
&batch_normalization_87/batchnorm/mul_2Mul/batch_normalization_87/moments/Squeeze:output:0(batch_normalization_87/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_87/batchnorm/mul_2?
/batch_normalization_87/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_87_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_87/batchnorm/ReadVariableOp?
$batch_normalization_87/batchnorm/subSub7batch_normalization_87/batchnorm/ReadVariableOp:value:0*batch_normalization_87/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_87/batchnorm/sub?
&batch_normalization_87/batchnorm/add_1AddV2*batch_normalization_87/batchnorm/mul_1:z:0(batch_normalization_87/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_87/batchnorm/add_1v
Relu_2Relu*batch_normalization_87/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu_2?
dense_73/MatMul/ReadVariableOpReadVariableOp'dense_73_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_73/MatMul/ReadVariableOp?
dense_73/MatMulMatMulRelu_2:activations:0&dense_73/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_73/MatMul?
5batch_normalization_88/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_88/moments/mean/reduction_indices?
#batch_normalization_88/moments/meanMeandense_73/MatMul:product:0>batch_normalization_88/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_88/moments/mean?
+batch_normalization_88/moments/StopGradientStopGradient,batch_normalization_88/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_88/moments/StopGradient?
0batch_normalization_88/moments/SquaredDifferenceSquaredDifferencedense_73/MatMul:product:04batch_normalization_88/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????22
0batch_normalization_88/moments/SquaredDifference?
9batch_normalization_88/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_88/moments/variance/reduction_indices?
'batch_normalization_88/moments/varianceMean4batch_normalization_88/moments/SquaredDifference:z:0Bbatch_normalization_88/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_88/moments/variance?
&batch_normalization_88/moments/SqueezeSqueeze,batch_normalization_88/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_88/moments/Squeeze?
(batch_normalization_88/moments/Squeeze_1Squeeze0batch_normalization_88/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_88/moments/Squeeze_1?
,batch_normalization_88/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_88/AssignMovingAvg/decay?
+batch_normalization_88/AssignMovingAvg/CastCast5batch_normalization_88/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_88/AssignMovingAvg/Cast?
5batch_normalization_88/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_88_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_88/AssignMovingAvg/ReadVariableOp?
*batch_normalization_88/AssignMovingAvg/subSub=batch_normalization_88/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_88/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_88/AssignMovingAvg/sub?
*batch_normalization_88/AssignMovingAvg/mulMul.batch_normalization_88/AssignMovingAvg/sub:z:0/batch_normalization_88/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_88/AssignMovingAvg/mul?
&batch_normalization_88/AssignMovingAvgAssignSubVariableOp>batch_normalization_88_assignmovingavg_readvariableop_resource.batch_normalization_88/AssignMovingAvg/mul:z:06^batch_normalization_88/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_88/AssignMovingAvg?
.batch_normalization_88/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_88/AssignMovingAvg_1/decay?
-batch_normalization_88/AssignMovingAvg_1/CastCast7batch_normalization_88/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_88/AssignMovingAvg_1/Cast?
7batch_normalization_88/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_88_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_88/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_88/AssignMovingAvg_1/subSub?batch_normalization_88/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_88/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_88/AssignMovingAvg_1/sub?
,batch_normalization_88/AssignMovingAvg_1/mulMul0batch_normalization_88/AssignMovingAvg_1/sub:z:01batch_normalization_88/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_88/AssignMovingAvg_1/mul?
(batch_normalization_88/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_88_assignmovingavg_1_readvariableop_resource0batch_normalization_88/AssignMovingAvg_1/mul:z:08^batch_normalization_88/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_88/AssignMovingAvg_1?
&batch_normalization_88/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2(
&batch_normalization_88/batchnorm/add/y?
$batch_normalization_88/batchnorm/addAddV21batch_normalization_88/moments/Squeeze_1:output:0/batch_normalization_88/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_88/batchnorm/add?
&batch_normalization_88/batchnorm/RsqrtRsqrt(batch_normalization_88/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_88/batchnorm/Rsqrt?
3batch_normalization_88/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_88_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_88/batchnorm/mul/ReadVariableOp?
$batch_normalization_88/batchnorm/mulMul*batch_normalization_88/batchnorm/Rsqrt:y:0;batch_normalization_88/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_88/batchnorm/mul?
&batch_normalization_88/batchnorm/mul_1Muldense_73/MatMul:product:0(batch_normalization_88/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_88/batchnorm/mul_1?
&batch_normalization_88/batchnorm/mul_2Mul/batch_normalization_88/moments/Squeeze:output:0(batch_normalization_88/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_88/batchnorm/mul_2?
/batch_normalization_88/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_88_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_88/batchnorm/ReadVariableOp?
$batch_normalization_88/batchnorm/subSub7batch_normalization_88/batchnorm/ReadVariableOp:value:0*batch_normalization_88/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_88/batchnorm/sub?
&batch_normalization_88/batchnorm/add_1AddV2*batch_normalization_88/batchnorm/mul_1:z:0(batch_normalization_88/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_88/batchnorm/add_1v
Relu_3Relu*batch_normalization_88/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu_3?
dense_74/MatMul/ReadVariableOpReadVariableOp'dense_74_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_74/MatMul/ReadVariableOp?
dense_74/MatMulMatMulRelu_3:activations:0&dense_74/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_74/MatMul?
dense_74/BiasAdd/ReadVariableOpReadVariableOp(dense_74_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_74/BiasAdd/ReadVariableOp?
dense_74/BiasAddBiasAdddense_74/MatMul:product:0'dense_74/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_74/BiasAddt
IdentityIdentitydense_74/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp'^batch_normalization_84/AssignMovingAvg6^batch_normalization_84/AssignMovingAvg/ReadVariableOp)^batch_normalization_84/AssignMovingAvg_18^batch_normalization_84/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_84/batchnorm/ReadVariableOp4^batch_normalization_84/batchnorm/mul/ReadVariableOp'^batch_normalization_85/AssignMovingAvg6^batch_normalization_85/AssignMovingAvg/ReadVariableOp)^batch_normalization_85/AssignMovingAvg_18^batch_normalization_85/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_85/batchnorm/ReadVariableOp4^batch_normalization_85/batchnorm/mul/ReadVariableOp'^batch_normalization_86/AssignMovingAvg6^batch_normalization_86/AssignMovingAvg/ReadVariableOp)^batch_normalization_86/AssignMovingAvg_18^batch_normalization_86/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_86/batchnorm/ReadVariableOp4^batch_normalization_86/batchnorm/mul/ReadVariableOp'^batch_normalization_87/AssignMovingAvg6^batch_normalization_87/AssignMovingAvg/ReadVariableOp)^batch_normalization_87/AssignMovingAvg_18^batch_normalization_87/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_87/batchnorm/ReadVariableOp4^batch_normalization_87/batchnorm/mul/ReadVariableOp'^batch_normalization_88/AssignMovingAvg6^batch_normalization_88/AssignMovingAvg/ReadVariableOp)^batch_normalization_88/AssignMovingAvg_18^batch_normalization_88/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_88/batchnorm/ReadVariableOp4^batch_normalization_88/batchnorm/mul/ReadVariableOp^dense_70/MatMul/ReadVariableOp^dense_71/MatMul/ReadVariableOp^dense_72/MatMul/ReadVariableOp^dense_73/MatMul/ReadVariableOp ^dense_74/BiasAdd/ReadVariableOp^dense_74/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????
: : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_84/AssignMovingAvg&batch_normalization_84/AssignMovingAvg2n
5batch_normalization_84/AssignMovingAvg/ReadVariableOp5batch_normalization_84/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_84/AssignMovingAvg_1(batch_normalization_84/AssignMovingAvg_12r
7batch_normalization_84/AssignMovingAvg_1/ReadVariableOp7batch_normalization_84/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_84/batchnorm/ReadVariableOp/batch_normalization_84/batchnorm/ReadVariableOp2j
3batch_normalization_84/batchnorm/mul/ReadVariableOp3batch_normalization_84/batchnorm/mul/ReadVariableOp2P
&batch_normalization_85/AssignMovingAvg&batch_normalization_85/AssignMovingAvg2n
5batch_normalization_85/AssignMovingAvg/ReadVariableOp5batch_normalization_85/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_85/AssignMovingAvg_1(batch_normalization_85/AssignMovingAvg_12r
7batch_normalization_85/AssignMovingAvg_1/ReadVariableOp7batch_normalization_85/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_85/batchnorm/ReadVariableOp/batch_normalization_85/batchnorm/ReadVariableOp2j
3batch_normalization_85/batchnorm/mul/ReadVariableOp3batch_normalization_85/batchnorm/mul/ReadVariableOp2P
&batch_normalization_86/AssignMovingAvg&batch_normalization_86/AssignMovingAvg2n
5batch_normalization_86/AssignMovingAvg/ReadVariableOp5batch_normalization_86/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_86/AssignMovingAvg_1(batch_normalization_86/AssignMovingAvg_12r
7batch_normalization_86/AssignMovingAvg_1/ReadVariableOp7batch_normalization_86/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_86/batchnorm/ReadVariableOp/batch_normalization_86/batchnorm/ReadVariableOp2j
3batch_normalization_86/batchnorm/mul/ReadVariableOp3batch_normalization_86/batchnorm/mul/ReadVariableOp2P
&batch_normalization_87/AssignMovingAvg&batch_normalization_87/AssignMovingAvg2n
5batch_normalization_87/AssignMovingAvg/ReadVariableOp5batch_normalization_87/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_87/AssignMovingAvg_1(batch_normalization_87/AssignMovingAvg_12r
7batch_normalization_87/AssignMovingAvg_1/ReadVariableOp7batch_normalization_87/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_87/batchnorm/ReadVariableOp/batch_normalization_87/batchnorm/ReadVariableOp2j
3batch_normalization_87/batchnorm/mul/ReadVariableOp3batch_normalization_87/batchnorm/mul/ReadVariableOp2P
&batch_normalization_88/AssignMovingAvg&batch_normalization_88/AssignMovingAvg2n
5batch_normalization_88/AssignMovingAvg/ReadVariableOp5batch_normalization_88/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_88/AssignMovingAvg_1(batch_normalization_88/AssignMovingAvg_12r
7batch_normalization_88/AssignMovingAvg_1/ReadVariableOp7batch_normalization_88/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_88/batchnorm/ReadVariableOp/batch_normalization_88/batchnorm/ReadVariableOp2j
3batch_normalization_88/batchnorm/mul/ReadVariableOp3batch_normalization_88/batchnorm/mul/ReadVariableOp2@
dense_70/MatMul/ReadVariableOpdense_70/MatMul/ReadVariableOp2@
dense_71/MatMul/ReadVariableOpdense_71/MatMul/ReadVariableOp2@
dense_72/MatMul/ReadVariableOpdense_72/MatMul/ReadVariableOp2@
dense_73/MatMul/ReadVariableOpdense_73/MatMul/ReadVariableOp2B
dense_74/BiasAdd/ReadVariableOpdense_74/BiasAdd/ReadVariableOp2@
dense_74/MatMul/ReadVariableOpdense_74/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????

!
_user_specified_name	input_1
?,
?
S__inference_batch_normalization_84_layer_call_and_return_conditional_losses_7060540

inputs5
'assignmovingavg_readvariableop_resource:
7
)assignmovingavg_1_readvariableop_resource:
3
%batchnorm_mul_readvariableop_resource:
/
!batchnorm_readvariableop_resource:

identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
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
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????
2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2
moments/Squeeze?
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
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg/Cast?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:
*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:
2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:
2
AssignMovingAvg/mul?
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
?#<2
AssignMovingAvg_1/decay?
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg_1/Cast?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:
*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:
2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:
2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1k
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2
batchnorm/add/y?
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
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????
2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:
2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????
2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?,
?
S__inference_batch_normalization_87_layer_call_and_return_conditional_losses_7063078

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
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
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
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
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg/Cast?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul?
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
?#<2
AssignMovingAvg_1/decay?
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg_1/Cast?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1k
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_85_layer_call_fn_7062858

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_85_layer_call_and_return_conditional_losses_70607062
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?,
?
S__inference_batch_normalization_87_layer_call_and_return_conditional_losses_7061038

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
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
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
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
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg/Cast?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul?
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
?#<2
AssignMovingAvg_1/decay?
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg_1/Cast?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1k
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0?????????
<
output_10
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:??
?
	bn_layers
dense_layers
regularization_losses
	variables
trainable_variables
	keras_api

signatures
?_default_save_signature
?__call__
+?&call_and_return_all_conditional_losses"
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
?
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
?
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
?
-layer_metrics
regularization_losses
.metrics
	variables
/layer_regularization_losses

0layers
1non_trainable_variables
trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?
2axis
	gamma
beta
moving_mean
moving_variance
3regularization_losses
4	variables
5trainable_variables
6	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
7axis
	gamma
beta
moving_mean
 moving_variance
8regularization_losses
9	variables
:trainable_variables
;	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
<axis
	gamma
beta
!moving_mean
"moving_variance
=regularization_losses
>	variables
?trainable_variables
@	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Aaxis
	gamma
beta
#moving_mean
$moving_variance
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Faxis
	gamma
beta
%moving_mean
&moving_variance
Gregularization_losses
H	variables
Itrainable_variables
J	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
(
K	keras_api"
_tf_keras_layer
?

'kernel
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

(kernel
Pregularization_losses
Q	variables
Rtrainable_variables
S	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

)kernel
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

*kernel
Xregularization_losses
Y	variables
Ztrainable_variables
[	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

+kernel
,bias
\regularization_losses
]	variables
^trainable_variables
_	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
T:R
2Fnonshared_model_1/feed_forward_sub_net_14/batch_normalization_84/gamma
S:Q
2Enonshared_model_1/feed_forward_sub_net_14/batch_normalization_84/beta
T:R2Fnonshared_model_1/feed_forward_sub_net_14/batch_normalization_85/gamma
S:Q2Enonshared_model_1/feed_forward_sub_net_14/batch_normalization_85/beta
T:R2Fnonshared_model_1/feed_forward_sub_net_14/batch_normalization_86/gamma
S:Q2Enonshared_model_1/feed_forward_sub_net_14/batch_normalization_86/beta
T:R2Fnonshared_model_1/feed_forward_sub_net_14/batch_normalization_87/gamma
S:Q2Enonshared_model_1/feed_forward_sub_net_14/batch_normalization_87/beta
T:R2Fnonshared_model_1/feed_forward_sub_net_14/batch_normalization_88/gamma
S:Q2Enonshared_model_1/feed_forward_sub_net_14/batch_normalization_88/beta
\:Z
 (2Lnonshared_model_1/feed_forward_sub_net_14/batch_normalization_84/moving_mean
`:^
 (2Pnonshared_model_1/feed_forward_sub_net_14/batch_normalization_84/moving_variance
\:Z (2Lnonshared_model_1/feed_forward_sub_net_14/batch_normalization_85/moving_mean
`:^ (2Pnonshared_model_1/feed_forward_sub_net_14/batch_normalization_85/moving_variance
\:Z (2Lnonshared_model_1/feed_forward_sub_net_14/batch_normalization_86/moving_mean
`:^ (2Pnonshared_model_1/feed_forward_sub_net_14/batch_normalization_86/moving_variance
\:Z (2Lnonshared_model_1/feed_forward_sub_net_14/batch_normalization_87/moving_mean
`:^ (2Pnonshared_model_1/feed_forward_sub_net_14/batch_normalization_87/moving_variance
\:Z (2Lnonshared_model_1/feed_forward_sub_net_14/batch_normalization_88/moving_mean
`:^ (2Pnonshared_model_1/feed_forward_sub_net_14/batch_normalization_88/moving_variance
K:I
29nonshared_model_1/feed_forward_sub_net_14/dense_70/kernel
K:I29nonshared_model_1/feed_forward_sub_net_14/dense_71/kernel
K:I29nonshared_model_1/feed_forward_sub_net_14/dense_72/kernel
K:I29nonshared_model_1/feed_forward_sub_net_14/dense_73/kernel
K:I
29nonshared_model_1/feed_forward_sub_net_14/dense_74/kernel
E:C
27nonshared_model_1/feed_forward_sub_net_14/dense_74/bias
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
?
`layer_metrics
3regularization_losses
ametrics
4	variables
blayer_regularization_losses

clayers
dnon_trainable_variables
5trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?
elayer_metrics
8regularization_losses
fmetrics
9	variables
glayer_regularization_losses

hlayers
inon_trainable_variables
:trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?
jlayer_metrics
=regularization_losses
kmetrics
>	variables
llayer_regularization_losses

mlayers
nnon_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?
olayer_metrics
Bregularization_losses
pmetrics
C	variables
qlayer_regularization_losses

rlayers
snon_trainable_variables
Dtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?
tlayer_metrics
Gregularization_losses
umetrics
H	variables
vlayer_regularization_losses

wlayers
xnon_trainable_variables
Itrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?
ylayer_metrics
Lregularization_losses
zmetrics
M	variables
{layer_regularization_losses

|layers
}non_trainable_variables
Ntrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
(0"
trackable_list_wrapper
'
(0"
trackable_list_wrapper
?
~layer_metrics
Pregularization_losses
metrics
Q	variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
Rtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
)0"
trackable_list_wrapper
'
)0"
trackable_list_wrapper
?
?layer_metrics
Tregularization_losses
?metrics
U	variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
Vtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
*0"
trackable_list_wrapper
'
*0"
trackable_list_wrapper
?
?layer_metrics
Xregularization_losses
?metrics
Y	variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
Ztrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?
?layer_metrics
\regularization_losses
?metrics
]	variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
^trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?B?
"__inference__wrapped_model_7060454input_1"?
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
9__inference_feed_forward_sub_net_14_layer_call_fn_7061995
9__inference_feed_forward_sub_net_14_layer_call_fn_7062052
9__inference_feed_forward_sub_net_14_layer_call_fn_7062109
9__inference_feed_forward_sub_net_14_layer_call_fn_7062166?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults? 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
T__inference_feed_forward_sub_net_14_layer_call_and_return_conditional_losses_7062272
T__inference_feed_forward_sub_net_14_layer_call_and_return_conditional_losses_7062458
T__inference_feed_forward_sub_net_14_layer_call_and_return_conditional_losses_7062564
T__inference_feed_forward_sub_net_14_layer_call_and_return_conditional_losses_7062750?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults? 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
%__inference_signature_wrapper_7061938input_1"?
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
?2?
8__inference_batch_normalization_84_layer_call_fn_7062763
8__inference_batch_normalization_84_layer_call_fn_7062776?
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
S__inference_batch_normalization_84_layer_call_and_return_conditional_losses_7062796
S__inference_batch_normalization_84_layer_call_and_return_conditional_losses_7062832?
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
8__inference_batch_normalization_85_layer_call_fn_7062845
8__inference_batch_normalization_85_layer_call_fn_7062858?
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
S__inference_batch_normalization_85_layer_call_and_return_conditional_losses_7062878
S__inference_batch_normalization_85_layer_call_and_return_conditional_losses_7062914?
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
8__inference_batch_normalization_86_layer_call_fn_7062927
8__inference_batch_normalization_86_layer_call_fn_7062940?
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
S__inference_batch_normalization_86_layer_call_and_return_conditional_losses_7062960
S__inference_batch_normalization_86_layer_call_and_return_conditional_losses_7062996?
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
8__inference_batch_normalization_87_layer_call_fn_7063009
8__inference_batch_normalization_87_layer_call_fn_7063022?
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
S__inference_batch_normalization_87_layer_call_and_return_conditional_losses_7063042
S__inference_batch_normalization_87_layer_call_and_return_conditional_losses_7063078?
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
8__inference_batch_normalization_88_layer_call_fn_7063091
8__inference_batch_normalization_88_layer_call_fn_7063104?
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
S__inference_batch_normalization_88_layer_call_and_return_conditional_losses_7063124
S__inference_batch_normalization_88_layer_call_and_return_conditional_losses_7063160?
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
*__inference_dense_70_layer_call_fn_7063167?
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
E__inference_dense_70_layer_call_and_return_conditional_losses_7063174?
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
*__inference_dense_71_layer_call_fn_7063181?
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
E__inference_dense_71_layer_call_and_return_conditional_losses_7063188?
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
*__inference_dense_72_layer_call_fn_7063195?
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
E__inference_dense_72_layer_call_and_return_conditional_losses_7063202?
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
*__inference_dense_73_layer_call_fn_7063209?
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
E__inference_dense_73_layer_call_and_return_conditional_losses_7063216?
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
*__inference_dense_74_layer_call_fn_7063225?
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
E__inference_dense_74_layer_call_and_return_conditional_losses_7063235?
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
"__inference__wrapped_model_7060454?' ("!)$#*&%+,0?-
&?#
!?
input_1?????????

? "3?0
.
output_1"?
output_1?????????
?
S__inference_batch_normalization_84_layer_call_and_return_conditional_losses_7062796b3?0
)?&
 ?
inputs?????????

p 
? "%?"
?
0?????????

? ?
S__inference_batch_normalization_84_layer_call_and_return_conditional_losses_7062832b3?0
)?&
 ?
inputs?????????

p
? "%?"
?
0?????????

? ?
8__inference_batch_normalization_84_layer_call_fn_7062763U3?0
)?&
 ?
inputs?????????

p 
? "??????????
?
8__inference_batch_normalization_84_layer_call_fn_7062776U3?0
)?&
 ?
inputs?????????

p
? "??????????
?
S__inference_batch_normalization_85_layer_call_and_return_conditional_losses_7062878b 3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
S__inference_batch_normalization_85_layer_call_and_return_conditional_losses_7062914b 3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? ?
8__inference_batch_normalization_85_layer_call_fn_7062845U 3?0
)?&
 ?
inputs?????????
p 
? "???????????
8__inference_batch_normalization_85_layer_call_fn_7062858U 3?0
)?&
 ?
inputs?????????
p
? "???????????
S__inference_batch_normalization_86_layer_call_and_return_conditional_losses_7062960b"!3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
S__inference_batch_normalization_86_layer_call_and_return_conditional_losses_7062996b!"3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? ?
8__inference_batch_normalization_86_layer_call_fn_7062927U"!3?0
)?&
 ?
inputs?????????
p 
? "???????????
8__inference_batch_normalization_86_layer_call_fn_7062940U!"3?0
)?&
 ?
inputs?????????
p
? "???????????
S__inference_batch_normalization_87_layer_call_and_return_conditional_losses_7063042b$#3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
S__inference_batch_normalization_87_layer_call_and_return_conditional_losses_7063078b#$3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? ?
8__inference_batch_normalization_87_layer_call_fn_7063009U$#3?0
)?&
 ?
inputs?????????
p 
? "???????????
8__inference_batch_normalization_87_layer_call_fn_7063022U#$3?0
)?&
 ?
inputs?????????
p
? "???????????
S__inference_batch_normalization_88_layer_call_and_return_conditional_losses_7063124b&%3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
S__inference_batch_normalization_88_layer_call_and_return_conditional_losses_7063160b%&3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? ?
8__inference_batch_normalization_88_layer_call_fn_7063091U&%3?0
)?&
 ?
inputs?????????
p 
? "???????????
8__inference_batch_normalization_88_layer_call_fn_7063104U%&3?0
)?&
 ?
inputs?????????
p
? "???????????
E__inference_dense_70_layer_call_and_return_conditional_losses_7063174['/?,
%?"
 ?
inputs?????????

? "%?"
?
0?????????
? |
*__inference_dense_70_layer_call_fn_7063167N'/?,
%?"
 ?
inputs?????????

? "???????????
E__inference_dense_71_layer_call_and_return_conditional_losses_7063188[(/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
*__inference_dense_71_layer_call_fn_7063181N(/?,
%?"
 ?
inputs?????????
? "???????????
E__inference_dense_72_layer_call_and_return_conditional_losses_7063202[)/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
*__inference_dense_72_layer_call_fn_7063195N)/?,
%?"
 ?
inputs?????????
? "???????????
E__inference_dense_73_layer_call_and_return_conditional_losses_7063216[*/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
*__inference_dense_73_layer_call_fn_7063209N*/?,
%?"
 ?
inputs?????????
? "???????????
E__inference_dense_74_layer_call_and_return_conditional_losses_7063235\+,/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????

? }
*__inference_dense_74_layer_call_fn_7063225O+,/?,
%?"
 ?
inputs?????????
? "??????????
?
T__inference_feed_forward_sub_net_14_layer_call_and_return_conditional_losses_7062272s' ("!)$#*&%+,.?+
$?!
?
x?????????

p 
? "%?"
?
0?????????

? ?
T__inference_feed_forward_sub_net_14_layer_call_and_return_conditional_losses_7062458s' (!")#$*%&+,.?+
$?!
?
x?????????

p
? "%?"
?
0?????????

? ?
T__inference_feed_forward_sub_net_14_layer_call_and_return_conditional_losses_7062564y' ("!)$#*&%+,4?1
*?'
!?
input_1?????????

p 
? "%?"
?
0?????????

? ?
T__inference_feed_forward_sub_net_14_layer_call_and_return_conditional_losses_7062750y' (!")#$*%&+,4?1
*?'
!?
input_1?????????

p
? "%?"
?
0?????????

? ?
9__inference_feed_forward_sub_net_14_layer_call_fn_7061995l' ("!)$#*&%+,4?1
*?'
!?
input_1?????????

p 
? "??????????
?
9__inference_feed_forward_sub_net_14_layer_call_fn_7062052f' ("!)$#*&%+,.?+
$?!
?
x?????????

p 
? "??????????
?
9__inference_feed_forward_sub_net_14_layer_call_fn_7062109f' (!")#$*%&+,.?+
$?!
?
x?????????

p
? "??????????
?
9__inference_feed_forward_sub_net_14_layer_call_fn_7062166l' (!")#$*%&+,4?1
*?'
!?
input_1?????????

p
? "??????????
?
%__inference_signature_wrapper_7061938?' ("!)$#*&%+,;?8
? 
1?.
,
input_1!?
input_1?????????
"3?0
.
output_1"?
output_1?????????
