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
Fnonshared_model_1/feed_forward_sub_net_12/batch_normalization_72/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*W
shared_nameHFnonshared_model_1/feed_forward_sub_net_12/batch_normalization_72/gamma
?
Znonshared_model_1/feed_forward_sub_net_12/batch_normalization_72/gamma/Read/ReadVariableOpReadVariableOpFnonshared_model_1/feed_forward_sub_net_12/batch_normalization_72/gamma*
_output_shapes
:
*
dtype0
?
Enonshared_model_1/feed_forward_sub_net_12/batch_normalization_72/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*V
shared_nameGEnonshared_model_1/feed_forward_sub_net_12/batch_normalization_72/beta
?
Ynonshared_model_1/feed_forward_sub_net_12/batch_normalization_72/beta/Read/ReadVariableOpReadVariableOpEnonshared_model_1/feed_forward_sub_net_12/batch_normalization_72/beta*
_output_shapes
:
*
dtype0
?
Fnonshared_model_1/feed_forward_sub_net_12/batch_normalization_73/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*W
shared_nameHFnonshared_model_1/feed_forward_sub_net_12/batch_normalization_73/gamma
?
Znonshared_model_1/feed_forward_sub_net_12/batch_normalization_73/gamma/Read/ReadVariableOpReadVariableOpFnonshared_model_1/feed_forward_sub_net_12/batch_normalization_73/gamma*
_output_shapes
:*
dtype0
?
Enonshared_model_1/feed_forward_sub_net_12/batch_normalization_73/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*V
shared_nameGEnonshared_model_1/feed_forward_sub_net_12/batch_normalization_73/beta
?
Ynonshared_model_1/feed_forward_sub_net_12/batch_normalization_73/beta/Read/ReadVariableOpReadVariableOpEnonshared_model_1/feed_forward_sub_net_12/batch_normalization_73/beta*
_output_shapes
:*
dtype0
?
Fnonshared_model_1/feed_forward_sub_net_12/batch_normalization_74/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*W
shared_nameHFnonshared_model_1/feed_forward_sub_net_12/batch_normalization_74/gamma
?
Znonshared_model_1/feed_forward_sub_net_12/batch_normalization_74/gamma/Read/ReadVariableOpReadVariableOpFnonshared_model_1/feed_forward_sub_net_12/batch_normalization_74/gamma*
_output_shapes
:*
dtype0
?
Enonshared_model_1/feed_forward_sub_net_12/batch_normalization_74/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*V
shared_nameGEnonshared_model_1/feed_forward_sub_net_12/batch_normalization_74/beta
?
Ynonshared_model_1/feed_forward_sub_net_12/batch_normalization_74/beta/Read/ReadVariableOpReadVariableOpEnonshared_model_1/feed_forward_sub_net_12/batch_normalization_74/beta*
_output_shapes
:*
dtype0
?
Fnonshared_model_1/feed_forward_sub_net_12/batch_normalization_75/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*W
shared_nameHFnonshared_model_1/feed_forward_sub_net_12/batch_normalization_75/gamma
?
Znonshared_model_1/feed_forward_sub_net_12/batch_normalization_75/gamma/Read/ReadVariableOpReadVariableOpFnonshared_model_1/feed_forward_sub_net_12/batch_normalization_75/gamma*
_output_shapes
:*
dtype0
?
Enonshared_model_1/feed_forward_sub_net_12/batch_normalization_75/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*V
shared_nameGEnonshared_model_1/feed_forward_sub_net_12/batch_normalization_75/beta
?
Ynonshared_model_1/feed_forward_sub_net_12/batch_normalization_75/beta/Read/ReadVariableOpReadVariableOpEnonshared_model_1/feed_forward_sub_net_12/batch_normalization_75/beta*
_output_shapes
:*
dtype0
?
Fnonshared_model_1/feed_forward_sub_net_12/batch_normalization_76/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*W
shared_nameHFnonshared_model_1/feed_forward_sub_net_12/batch_normalization_76/gamma
?
Znonshared_model_1/feed_forward_sub_net_12/batch_normalization_76/gamma/Read/ReadVariableOpReadVariableOpFnonshared_model_1/feed_forward_sub_net_12/batch_normalization_76/gamma*
_output_shapes
:*
dtype0
?
Enonshared_model_1/feed_forward_sub_net_12/batch_normalization_76/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*V
shared_nameGEnonshared_model_1/feed_forward_sub_net_12/batch_normalization_76/beta
?
Ynonshared_model_1/feed_forward_sub_net_12/batch_normalization_76/beta/Read/ReadVariableOpReadVariableOpEnonshared_model_1/feed_forward_sub_net_12/batch_normalization_76/beta*
_output_shapes
:*
dtype0
?
Lnonshared_model_1/feed_forward_sub_net_12/batch_normalization_72/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*]
shared_nameNLnonshared_model_1/feed_forward_sub_net_12/batch_normalization_72/moving_mean
?
`nonshared_model_1/feed_forward_sub_net_12/batch_normalization_72/moving_mean/Read/ReadVariableOpReadVariableOpLnonshared_model_1/feed_forward_sub_net_12/batch_normalization_72/moving_mean*
_output_shapes
:
*
dtype0
?
Pnonshared_model_1/feed_forward_sub_net_12/batch_normalization_72/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*a
shared_nameRPnonshared_model_1/feed_forward_sub_net_12/batch_normalization_72/moving_variance
?
dnonshared_model_1/feed_forward_sub_net_12/batch_normalization_72/moving_variance/Read/ReadVariableOpReadVariableOpPnonshared_model_1/feed_forward_sub_net_12/batch_normalization_72/moving_variance*
_output_shapes
:
*
dtype0
?
Lnonshared_model_1/feed_forward_sub_net_12/batch_normalization_73/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*]
shared_nameNLnonshared_model_1/feed_forward_sub_net_12/batch_normalization_73/moving_mean
?
`nonshared_model_1/feed_forward_sub_net_12/batch_normalization_73/moving_mean/Read/ReadVariableOpReadVariableOpLnonshared_model_1/feed_forward_sub_net_12/batch_normalization_73/moving_mean*
_output_shapes
:*
dtype0
?
Pnonshared_model_1/feed_forward_sub_net_12/batch_normalization_73/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*a
shared_nameRPnonshared_model_1/feed_forward_sub_net_12/batch_normalization_73/moving_variance
?
dnonshared_model_1/feed_forward_sub_net_12/batch_normalization_73/moving_variance/Read/ReadVariableOpReadVariableOpPnonshared_model_1/feed_forward_sub_net_12/batch_normalization_73/moving_variance*
_output_shapes
:*
dtype0
?
Lnonshared_model_1/feed_forward_sub_net_12/batch_normalization_74/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*]
shared_nameNLnonshared_model_1/feed_forward_sub_net_12/batch_normalization_74/moving_mean
?
`nonshared_model_1/feed_forward_sub_net_12/batch_normalization_74/moving_mean/Read/ReadVariableOpReadVariableOpLnonshared_model_1/feed_forward_sub_net_12/batch_normalization_74/moving_mean*
_output_shapes
:*
dtype0
?
Pnonshared_model_1/feed_forward_sub_net_12/batch_normalization_74/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*a
shared_nameRPnonshared_model_1/feed_forward_sub_net_12/batch_normalization_74/moving_variance
?
dnonshared_model_1/feed_forward_sub_net_12/batch_normalization_74/moving_variance/Read/ReadVariableOpReadVariableOpPnonshared_model_1/feed_forward_sub_net_12/batch_normalization_74/moving_variance*
_output_shapes
:*
dtype0
?
Lnonshared_model_1/feed_forward_sub_net_12/batch_normalization_75/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*]
shared_nameNLnonshared_model_1/feed_forward_sub_net_12/batch_normalization_75/moving_mean
?
`nonshared_model_1/feed_forward_sub_net_12/batch_normalization_75/moving_mean/Read/ReadVariableOpReadVariableOpLnonshared_model_1/feed_forward_sub_net_12/batch_normalization_75/moving_mean*
_output_shapes
:*
dtype0
?
Pnonshared_model_1/feed_forward_sub_net_12/batch_normalization_75/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*a
shared_nameRPnonshared_model_1/feed_forward_sub_net_12/batch_normalization_75/moving_variance
?
dnonshared_model_1/feed_forward_sub_net_12/batch_normalization_75/moving_variance/Read/ReadVariableOpReadVariableOpPnonshared_model_1/feed_forward_sub_net_12/batch_normalization_75/moving_variance*
_output_shapes
:*
dtype0
?
Lnonshared_model_1/feed_forward_sub_net_12/batch_normalization_76/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*]
shared_nameNLnonshared_model_1/feed_forward_sub_net_12/batch_normalization_76/moving_mean
?
`nonshared_model_1/feed_forward_sub_net_12/batch_normalization_76/moving_mean/Read/ReadVariableOpReadVariableOpLnonshared_model_1/feed_forward_sub_net_12/batch_normalization_76/moving_mean*
_output_shapes
:*
dtype0
?
Pnonshared_model_1/feed_forward_sub_net_12/batch_normalization_76/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*a
shared_nameRPnonshared_model_1/feed_forward_sub_net_12/batch_normalization_76/moving_variance
?
dnonshared_model_1/feed_forward_sub_net_12/batch_normalization_76/moving_variance/Read/ReadVariableOpReadVariableOpPnonshared_model_1/feed_forward_sub_net_12/batch_normalization_76/moving_variance*
_output_shapes
:*
dtype0
?
9nonshared_model_1/feed_forward_sub_net_12/dense_60/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*J
shared_name;9nonshared_model_1/feed_forward_sub_net_12/dense_60/kernel
?
Mnonshared_model_1/feed_forward_sub_net_12/dense_60/kernel/Read/ReadVariableOpReadVariableOp9nonshared_model_1/feed_forward_sub_net_12/dense_60/kernel*
_output_shapes

:
*
dtype0
?
9nonshared_model_1/feed_forward_sub_net_12/dense_61/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*J
shared_name;9nonshared_model_1/feed_forward_sub_net_12/dense_61/kernel
?
Mnonshared_model_1/feed_forward_sub_net_12/dense_61/kernel/Read/ReadVariableOpReadVariableOp9nonshared_model_1/feed_forward_sub_net_12/dense_61/kernel*
_output_shapes

:*
dtype0
?
9nonshared_model_1/feed_forward_sub_net_12/dense_62/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*J
shared_name;9nonshared_model_1/feed_forward_sub_net_12/dense_62/kernel
?
Mnonshared_model_1/feed_forward_sub_net_12/dense_62/kernel/Read/ReadVariableOpReadVariableOp9nonshared_model_1/feed_forward_sub_net_12/dense_62/kernel*
_output_shapes

:*
dtype0
?
9nonshared_model_1/feed_forward_sub_net_12/dense_63/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*J
shared_name;9nonshared_model_1/feed_forward_sub_net_12/dense_63/kernel
?
Mnonshared_model_1/feed_forward_sub_net_12/dense_63/kernel/Read/ReadVariableOpReadVariableOp9nonshared_model_1/feed_forward_sub_net_12/dense_63/kernel*
_output_shapes

:*
dtype0
?
9nonshared_model_1/feed_forward_sub_net_12/dense_64/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*J
shared_name;9nonshared_model_1/feed_forward_sub_net_12/dense_64/kernel
?
Mnonshared_model_1/feed_forward_sub_net_12/dense_64/kernel/Read/ReadVariableOpReadVariableOp9nonshared_model_1/feed_forward_sub_net_12/dense_64/kernel*
_output_shapes

:
*
dtype0
?
7nonshared_model_1/feed_forward_sub_net_12/dense_64/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*H
shared_name97nonshared_model_1/feed_forward_sub_net_12/dense_64/bias
?
Knonshared_model_1/feed_forward_sub_net_12/dense_64/bias/Read/ReadVariableOpReadVariableOp7nonshared_model_1/feed_forward_sub_net_12/dense_64/bias*
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
VARIABLE_VALUEFnonshared_model_1/feed_forward_sub_net_12/batch_normalization_72/gamma&variables/0/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEEnonshared_model_1/feed_forward_sub_net_12/batch_normalization_72/beta&variables/1/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEFnonshared_model_1/feed_forward_sub_net_12/batch_normalization_73/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEEnonshared_model_1/feed_forward_sub_net_12/batch_normalization_73/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEFnonshared_model_1/feed_forward_sub_net_12/batch_normalization_74/gamma&variables/4/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEEnonshared_model_1/feed_forward_sub_net_12/batch_normalization_74/beta&variables/5/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEFnonshared_model_1/feed_forward_sub_net_12/batch_normalization_75/gamma&variables/6/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEEnonshared_model_1/feed_forward_sub_net_12/batch_normalization_75/beta&variables/7/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEFnonshared_model_1/feed_forward_sub_net_12/batch_normalization_76/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEEnonshared_model_1/feed_forward_sub_net_12/batch_normalization_76/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUELnonshared_model_1/feed_forward_sub_net_12/batch_normalization_72/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEPnonshared_model_1/feed_forward_sub_net_12/batch_normalization_72/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUELnonshared_model_1/feed_forward_sub_net_12/batch_normalization_73/moving_mean'variables/12/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEPnonshared_model_1/feed_forward_sub_net_12/batch_normalization_73/moving_variance'variables/13/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUELnonshared_model_1/feed_forward_sub_net_12/batch_normalization_74/moving_mean'variables/14/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEPnonshared_model_1/feed_forward_sub_net_12/batch_normalization_74/moving_variance'variables/15/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUELnonshared_model_1/feed_forward_sub_net_12/batch_normalization_75/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEPnonshared_model_1/feed_forward_sub_net_12/batch_normalization_75/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUELnonshared_model_1/feed_forward_sub_net_12/batch_normalization_76/moving_mean'variables/18/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEPnonshared_model_1/feed_forward_sub_net_12/batch_normalization_76/moving_variance'variables/19/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9nonshared_model_1/feed_forward_sub_net_12/dense_60/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9nonshared_model_1/feed_forward_sub_net_12/dense_61/kernel'variables/21/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9nonshared_model_1/feed_forward_sub_net_12/dense_62/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9nonshared_model_1/feed_forward_sub_net_12/dense_63/kernel'variables/23/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9nonshared_model_1/feed_forward_sub_net_12/dense_64/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE7nonshared_model_1/feed_forward_sub_net_12/dense_64/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1Pnonshared_model_1/feed_forward_sub_net_12/batch_normalization_72/moving_varianceFnonshared_model_1/feed_forward_sub_net_12/batch_normalization_72/gammaLnonshared_model_1/feed_forward_sub_net_12/batch_normalization_72/moving_meanEnonshared_model_1/feed_forward_sub_net_12/batch_normalization_72/beta9nonshared_model_1/feed_forward_sub_net_12/dense_60/kernelPnonshared_model_1/feed_forward_sub_net_12/batch_normalization_73/moving_varianceFnonshared_model_1/feed_forward_sub_net_12/batch_normalization_73/gammaLnonshared_model_1/feed_forward_sub_net_12/batch_normalization_73/moving_meanEnonshared_model_1/feed_forward_sub_net_12/batch_normalization_73/beta9nonshared_model_1/feed_forward_sub_net_12/dense_61/kernelPnonshared_model_1/feed_forward_sub_net_12/batch_normalization_74/moving_varianceFnonshared_model_1/feed_forward_sub_net_12/batch_normalization_74/gammaLnonshared_model_1/feed_forward_sub_net_12/batch_normalization_74/moving_meanEnonshared_model_1/feed_forward_sub_net_12/batch_normalization_74/beta9nonshared_model_1/feed_forward_sub_net_12/dense_62/kernelPnonshared_model_1/feed_forward_sub_net_12/batch_normalization_75/moving_varianceFnonshared_model_1/feed_forward_sub_net_12/batch_normalization_75/gammaLnonshared_model_1/feed_forward_sub_net_12/batch_normalization_75/moving_meanEnonshared_model_1/feed_forward_sub_net_12/batch_normalization_75/beta9nonshared_model_1/feed_forward_sub_net_12/dense_63/kernelPnonshared_model_1/feed_forward_sub_net_12/batch_normalization_76/moving_varianceFnonshared_model_1/feed_forward_sub_net_12/batch_normalization_76/gammaLnonshared_model_1/feed_forward_sub_net_12/batch_normalization_76/moving_meanEnonshared_model_1/feed_forward_sub_net_12/batch_normalization_76/beta9nonshared_model_1/feed_forward_sub_net_12/dense_64/kernel7nonshared_model_1/feed_forward_sub_net_12/dense_64/bias*&
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
%__inference_signature_wrapper_7055546
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameZnonshared_model_1/feed_forward_sub_net_12/batch_normalization_72/gamma/Read/ReadVariableOpYnonshared_model_1/feed_forward_sub_net_12/batch_normalization_72/beta/Read/ReadVariableOpZnonshared_model_1/feed_forward_sub_net_12/batch_normalization_73/gamma/Read/ReadVariableOpYnonshared_model_1/feed_forward_sub_net_12/batch_normalization_73/beta/Read/ReadVariableOpZnonshared_model_1/feed_forward_sub_net_12/batch_normalization_74/gamma/Read/ReadVariableOpYnonshared_model_1/feed_forward_sub_net_12/batch_normalization_74/beta/Read/ReadVariableOpZnonshared_model_1/feed_forward_sub_net_12/batch_normalization_75/gamma/Read/ReadVariableOpYnonshared_model_1/feed_forward_sub_net_12/batch_normalization_75/beta/Read/ReadVariableOpZnonshared_model_1/feed_forward_sub_net_12/batch_normalization_76/gamma/Read/ReadVariableOpYnonshared_model_1/feed_forward_sub_net_12/batch_normalization_76/beta/Read/ReadVariableOp`nonshared_model_1/feed_forward_sub_net_12/batch_normalization_72/moving_mean/Read/ReadVariableOpdnonshared_model_1/feed_forward_sub_net_12/batch_normalization_72/moving_variance/Read/ReadVariableOp`nonshared_model_1/feed_forward_sub_net_12/batch_normalization_73/moving_mean/Read/ReadVariableOpdnonshared_model_1/feed_forward_sub_net_12/batch_normalization_73/moving_variance/Read/ReadVariableOp`nonshared_model_1/feed_forward_sub_net_12/batch_normalization_74/moving_mean/Read/ReadVariableOpdnonshared_model_1/feed_forward_sub_net_12/batch_normalization_74/moving_variance/Read/ReadVariableOp`nonshared_model_1/feed_forward_sub_net_12/batch_normalization_75/moving_mean/Read/ReadVariableOpdnonshared_model_1/feed_forward_sub_net_12/batch_normalization_75/moving_variance/Read/ReadVariableOp`nonshared_model_1/feed_forward_sub_net_12/batch_normalization_76/moving_mean/Read/ReadVariableOpdnonshared_model_1/feed_forward_sub_net_12/batch_normalization_76/moving_variance/Read/ReadVariableOpMnonshared_model_1/feed_forward_sub_net_12/dense_60/kernel/Read/ReadVariableOpMnonshared_model_1/feed_forward_sub_net_12/dense_61/kernel/Read/ReadVariableOpMnonshared_model_1/feed_forward_sub_net_12/dense_62/kernel/Read/ReadVariableOpMnonshared_model_1/feed_forward_sub_net_12/dense_63/kernel/Read/ReadVariableOpMnonshared_model_1/feed_forward_sub_net_12/dense_64/kernel/Read/ReadVariableOpKnonshared_model_1/feed_forward_sub_net_12/dense_64/bias/Read/ReadVariableOpConst*'
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
 __inference__traced_save_7056944
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameFnonshared_model_1/feed_forward_sub_net_12/batch_normalization_72/gammaEnonshared_model_1/feed_forward_sub_net_12/batch_normalization_72/betaFnonshared_model_1/feed_forward_sub_net_12/batch_normalization_73/gammaEnonshared_model_1/feed_forward_sub_net_12/batch_normalization_73/betaFnonshared_model_1/feed_forward_sub_net_12/batch_normalization_74/gammaEnonshared_model_1/feed_forward_sub_net_12/batch_normalization_74/betaFnonshared_model_1/feed_forward_sub_net_12/batch_normalization_75/gammaEnonshared_model_1/feed_forward_sub_net_12/batch_normalization_75/betaFnonshared_model_1/feed_forward_sub_net_12/batch_normalization_76/gammaEnonshared_model_1/feed_forward_sub_net_12/batch_normalization_76/betaLnonshared_model_1/feed_forward_sub_net_12/batch_normalization_72/moving_meanPnonshared_model_1/feed_forward_sub_net_12/batch_normalization_72/moving_varianceLnonshared_model_1/feed_forward_sub_net_12/batch_normalization_73/moving_meanPnonshared_model_1/feed_forward_sub_net_12/batch_normalization_73/moving_varianceLnonshared_model_1/feed_forward_sub_net_12/batch_normalization_74/moving_meanPnonshared_model_1/feed_forward_sub_net_12/batch_normalization_74/moving_varianceLnonshared_model_1/feed_forward_sub_net_12/batch_normalization_75/moving_meanPnonshared_model_1/feed_forward_sub_net_12/batch_normalization_75/moving_varianceLnonshared_model_1/feed_forward_sub_net_12/batch_normalization_76/moving_meanPnonshared_model_1/feed_forward_sub_net_12/batch_normalization_76/moving_variance9nonshared_model_1/feed_forward_sub_net_12/dense_60/kernel9nonshared_model_1/feed_forward_sub_net_12/dense_61/kernel9nonshared_model_1/feed_forward_sub_net_12/dense_62/kernel9nonshared_model_1/feed_forward_sub_net_12/dense_63/kernel9nonshared_model_1/feed_forward_sub_net_12/dense_64/kernel7nonshared_model_1/feed_forward_sub_net_12/dense_64/bias*&
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
#__inference__traced_restore_7057032??
??
?"
"__inference__wrapped_model_7054062
input_1^
Pfeed_forward_sub_net_12_batch_normalization_72_batchnorm_readvariableop_resource:
b
Tfeed_forward_sub_net_12_batch_normalization_72_batchnorm_mul_readvariableop_resource:
`
Rfeed_forward_sub_net_12_batch_normalization_72_batchnorm_readvariableop_1_resource:
`
Rfeed_forward_sub_net_12_batch_normalization_72_batchnorm_readvariableop_2_resource:
Q
?feed_forward_sub_net_12_dense_60_matmul_readvariableop_resource:
^
Pfeed_forward_sub_net_12_batch_normalization_73_batchnorm_readvariableop_resource:b
Tfeed_forward_sub_net_12_batch_normalization_73_batchnorm_mul_readvariableop_resource:`
Rfeed_forward_sub_net_12_batch_normalization_73_batchnorm_readvariableop_1_resource:`
Rfeed_forward_sub_net_12_batch_normalization_73_batchnorm_readvariableop_2_resource:Q
?feed_forward_sub_net_12_dense_61_matmul_readvariableop_resource:^
Pfeed_forward_sub_net_12_batch_normalization_74_batchnorm_readvariableop_resource:b
Tfeed_forward_sub_net_12_batch_normalization_74_batchnorm_mul_readvariableop_resource:`
Rfeed_forward_sub_net_12_batch_normalization_74_batchnorm_readvariableop_1_resource:`
Rfeed_forward_sub_net_12_batch_normalization_74_batchnorm_readvariableop_2_resource:Q
?feed_forward_sub_net_12_dense_62_matmul_readvariableop_resource:^
Pfeed_forward_sub_net_12_batch_normalization_75_batchnorm_readvariableop_resource:b
Tfeed_forward_sub_net_12_batch_normalization_75_batchnorm_mul_readvariableop_resource:`
Rfeed_forward_sub_net_12_batch_normalization_75_batchnorm_readvariableop_1_resource:`
Rfeed_forward_sub_net_12_batch_normalization_75_batchnorm_readvariableop_2_resource:Q
?feed_forward_sub_net_12_dense_63_matmul_readvariableop_resource:^
Pfeed_forward_sub_net_12_batch_normalization_76_batchnorm_readvariableop_resource:b
Tfeed_forward_sub_net_12_batch_normalization_76_batchnorm_mul_readvariableop_resource:`
Rfeed_forward_sub_net_12_batch_normalization_76_batchnorm_readvariableop_1_resource:`
Rfeed_forward_sub_net_12_batch_normalization_76_batchnorm_readvariableop_2_resource:Q
?feed_forward_sub_net_12_dense_64_matmul_readvariableop_resource:
N
@feed_forward_sub_net_12_dense_64_biasadd_readvariableop_resource:

identity??Gfeed_forward_sub_net_12/batch_normalization_72/batchnorm/ReadVariableOp?Ifeed_forward_sub_net_12/batch_normalization_72/batchnorm/ReadVariableOp_1?Ifeed_forward_sub_net_12/batch_normalization_72/batchnorm/ReadVariableOp_2?Kfeed_forward_sub_net_12/batch_normalization_72/batchnorm/mul/ReadVariableOp?Gfeed_forward_sub_net_12/batch_normalization_73/batchnorm/ReadVariableOp?Ifeed_forward_sub_net_12/batch_normalization_73/batchnorm/ReadVariableOp_1?Ifeed_forward_sub_net_12/batch_normalization_73/batchnorm/ReadVariableOp_2?Kfeed_forward_sub_net_12/batch_normalization_73/batchnorm/mul/ReadVariableOp?Gfeed_forward_sub_net_12/batch_normalization_74/batchnorm/ReadVariableOp?Ifeed_forward_sub_net_12/batch_normalization_74/batchnorm/ReadVariableOp_1?Ifeed_forward_sub_net_12/batch_normalization_74/batchnorm/ReadVariableOp_2?Kfeed_forward_sub_net_12/batch_normalization_74/batchnorm/mul/ReadVariableOp?Gfeed_forward_sub_net_12/batch_normalization_75/batchnorm/ReadVariableOp?Ifeed_forward_sub_net_12/batch_normalization_75/batchnorm/ReadVariableOp_1?Ifeed_forward_sub_net_12/batch_normalization_75/batchnorm/ReadVariableOp_2?Kfeed_forward_sub_net_12/batch_normalization_75/batchnorm/mul/ReadVariableOp?Gfeed_forward_sub_net_12/batch_normalization_76/batchnorm/ReadVariableOp?Ifeed_forward_sub_net_12/batch_normalization_76/batchnorm/ReadVariableOp_1?Ifeed_forward_sub_net_12/batch_normalization_76/batchnorm/ReadVariableOp_2?Kfeed_forward_sub_net_12/batch_normalization_76/batchnorm/mul/ReadVariableOp?6feed_forward_sub_net_12/dense_60/MatMul/ReadVariableOp?6feed_forward_sub_net_12/dense_61/MatMul/ReadVariableOp?6feed_forward_sub_net_12/dense_62/MatMul/ReadVariableOp?6feed_forward_sub_net_12/dense_63/MatMul/ReadVariableOp?7feed_forward_sub_net_12/dense_64/BiasAdd/ReadVariableOp?6feed_forward_sub_net_12/dense_64/MatMul/ReadVariableOp?
Gfeed_forward_sub_net_12/batch_normalization_72/batchnorm/ReadVariableOpReadVariableOpPfeed_forward_sub_net_12_batch_normalization_72_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype02I
Gfeed_forward_sub_net_12/batch_normalization_72/batchnorm/ReadVariableOp?
>feed_forward_sub_net_12/batch_normalization_72/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2@
>feed_forward_sub_net_12/batch_normalization_72/batchnorm/add/y?
<feed_forward_sub_net_12/batch_normalization_72/batchnorm/addAddV2Ofeed_forward_sub_net_12/batch_normalization_72/batchnorm/ReadVariableOp:value:0Gfeed_forward_sub_net_12/batch_normalization_72/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2>
<feed_forward_sub_net_12/batch_normalization_72/batchnorm/add?
>feed_forward_sub_net_12/batch_normalization_72/batchnorm/RsqrtRsqrt@feed_forward_sub_net_12/batch_normalization_72/batchnorm/add:z:0*
T0*
_output_shapes
:
2@
>feed_forward_sub_net_12/batch_normalization_72/batchnorm/Rsqrt?
Kfeed_forward_sub_net_12/batch_normalization_72/batchnorm/mul/ReadVariableOpReadVariableOpTfeed_forward_sub_net_12_batch_normalization_72_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype02M
Kfeed_forward_sub_net_12/batch_normalization_72/batchnorm/mul/ReadVariableOp?
<feed_forward_sub_net_12/batch_normalization_72/batchnorm/mulMulBfeed_forward_sub_net_12/batch_normalization_72/batchnorm/Rsqrt:y:0Sfeed_forward_sub_net_12/batch_normalization_72/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2>
<feed_forward_sub_net_12/batch_normalization_72/batchnorm/mul?
>feed_forward_sub_net_12/batch_normalization_72/batchnorm/mul_1Mulinput_1@feed_forward_sub_net_12/batch_normalization_72/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????
2@
>feed_forward_sub_net_12/batch_normalization_72/batchnorm/mul_1?
Ifeed_forward_sub_net_12/batch_normalization_72/batchnorm/ReadVariableOp_1ReadVariableOpRfeed_forward_sub_net_12_batch_normalization_72_batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype02K
Ifeed_forward_sub_net_12/batch_normalization_72/batchnorm/ReadVariableOp_1?
>feed_forward_sub_net_12/batch_normalization_72/batchnorm/mul_2MulQfeed_forward_sub_net_12/batch_normalization_72/batchnorm/ReadVariableOp_1:value:0@feed_forward_sub_net_12/batch_normalization_72/batchnorm/mul:z:0*
T0*
_output_shapes
:
2@
>feed_forward_sub_net_12/batch_normalization_72/batchnorm/mul_2?
Ifeed_forward_sub_net_12/batch_normalization_72/batchnorm/ReadVariableOp_2ReadVariableOpRfeed_forward_sub_net_12_batch_normalization_72_batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype02K
Ifeed_forward_sub_net_12/batch_normalization_72/batchnorm/ReadVariableOp_2?
<feed_forward_sub_net_12/batch_normalization_72/batchnorm/subSubQfeed_forward_sub_net_12/batch_normalization_72/batchnorm/ReadVariableOp_2:value:0Bfeed_forward_sub_net_12/batch_normalization_72/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2>
<feed_forward_sub_net_12/batch_normalization_72/batchnorm/sub?
>feed_forward_sub_net_12/batch_normalization_72/batchnorm/add_1AddV2Bfeed_forward_sub_net_12/batch_normalization_72/batchnorm/mul_1:z:0@feed_forward_sub_net_12/batch_normalization_72/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????
2@
>feed_forward_sub_net_12/batch_normalization_72/batchnorm/add_1?
6feed_forward_sub_net_12/dense_60/MatMul/ReadVariableOpReadVariableOp?feed_forward_sub_net_12_dense_60_matmul_readvariableop_resource*
_output_shapes

:
*
dtype028
6feed_forward_sub_net_12/dense_60/MatMul/ReadVariableOp?
'feed_forward_sub_net_12/dense_60/MatMulMatMulBfeed_forward_sub_net_12/batch_normalization_72/batchnorm/add_1:z:0>feed_forward_sub_net_12/dense_60/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2)
'feed_forward_sub_net_12/dense_60/MatMul?
Gfeed_forward_sub_net_12/batch_normalization_73/batchnorm/ReadVariableOpReadVariableOpPfeed_forward_sub_net_12_batch_normalization_73_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02I
Gfeed_forward_sub_net_12/batch_normalization_73/batchnorm/ReadVariableOp?
>feed_forward_sub_net_12/batch_normalization_73/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2@
>feed_forward_sub_net_12/batch_normalization_73/batchnorm/add/y?
<feed_forward_sub_net_12/batch_normalization_73/batchnorm/addAddV2Ofeed_forward_sub_net_12/batch_normalization_73/batchnorm/ReadVariableOp:value:0Gfeed_forward_sub_net_12/batch_normalization_73/batchnorm/add/y:output:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_12/batch_normalization_73/batchnorm/add?
>feed_forward_sub_net_12/batch_normalization_73/batchnorm/RsqrtRsqrt@feed_forward_sub_net_12/batch_normalization_73/batchnorm/add:z:0*
T0*
_output_shapes
:2@
>feed_forward_sub_net_12/batch_normalization_73/batchnorm/Rsqrt?
Kfeed_forward_sub_net_12/batch_normalization_73/batchnorm/mul/ReadVariableOpReadVariableOpTfeed_forward_sub_net_12_batch_normalization_73_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02M
Kfeed_forward_sub_net_12/batch_normalization_73/batchnorm/mul/ReadVariableOp?
<feed_forward_sub_net_12/batch_normalization_73/batchnorm/mulMulBfeed_forward_sub_net_12/batch_normalization_73/batchnorm/Rsqrt:y:0Sfeed_forward_sub_net_12/batch_normalization_73/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_12/batch_normalization_73/batchnorm/mul?
>feed_forward_sub_net_12/batch_normalization_73/batchnorm/mul_1Mul1feed_forward_sub_net_12/dense_60/MatMul:product:0@feed_forward_sub_net_12/batch_normalization_73/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2@
>feed_forward_sub_net_12/batch_normalization_73/batchnorm/mul_1?
Ifeed_forward_sub_net_12/batch_normalization_73/batchnorm/ReadVariableOp_1ReadVariableOpRfeed_forward_sub_net_12_batch_normalization_73_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02K
Ifeed_forward_sub_net_12/batch_normalization_73/batchnorm/ReadVariableOp_1?
>feed_forward_sub_net_12/batch_normalization_73/batchnorm/mul_2MulQfeed_forward_sub_net_12/batch_normalization_73/batchnorm/ReadVariableOp_1:value:0@feed_forward_sub_net_12/batch_normalization_73/batchnorm/mul:z:0*
T0*
_output_shapes
:2@
>feed_forward_sub_net_12/batch_normalization_73/batchnorm/mul_2?
Ifeed_forward_sub_net_12/batch_normalization_73/batchnorm/ReadVariableOp_2ReadVariableOpRfeed_forward_sub_net_12_batch_normalization_73_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02K
Ifeed_forward_sub_net_12/batch_normalization_73/batchnorm/ReadVariableOp_2?
<feed_forward_sub_net_12/batch_normalization_73/batchnorm/subSubQfeed_forward_sub_net_12/batch_normalization_73/batchnorm/ReadVariableOp_2:value:0Bfeed_forward_sub_net_12/batch_normalization_73/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_12/batch_normalization_73/batchnorm/sub?
>feed_forward_sub_net_12/batch_normalization_73/batchnorm/add_1AddV2Bfeed_forward_sub_net_12/batch_normalization_73/batchnorm/mul_1:z:0@feed_forward_sub_net_12/batch_normalization_73/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2@
>feed_forward_sub_net_12/batch_normalization_73/batchnorm/add_1?
feed_forward_sub_net_12/ReluReluBfeed_forward_sub_net_12/batch_normalization_73/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
feed_forward_sub_net_12/Relu?
6feed_forward_sub_net_12/dense_61/MatMul/ReadVariableOpReadVariableOp?feed_forward_sub_net_12_dense_61_matmul_readvariableop_resource*
_output_shapes

:*
dtype028
6feed_forward_sub_net_12/dense_61/MatMul/ReadVariableOp?
'feed_forward_sub_net_12/dense_61/MatMulMatMul*feed_forward_sub_net_12/Relu:activations:0>feed_forward_sub_net_12/dense_61/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2)
'feed_forward_sub_net_12/dense_61/MatMul?
Gfeed_forward_sub_net_12/batch_normalization_74/batchnorm/ReadVariableOpReadVariableOpPfeed_forward_sub_net_12_batch_normalization_74_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02I
Gfeed_forward_sub_net_12/batch_normalization_74/batchnorm/ReadVariableOp?
>feed_forward_sub_net_12/batch_normalization_74/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2@
>feed_forward_sub_net_12/batch_normalization_74/batchnorm/add/y?
<feed_forward_sub_net_12/batch_normalization_74/batchnorm/addAddV2Ofeed_forward_sub_net_12/batch_normalization_74/batchnorm/ReadVariableOp:value:0Gfeed_forward_sub_net_12/batch_normalization_74/batchnorm/add/y:output:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_12/batch_normalization_74/batchnorm/add?
>feed_forward_sub_net_12/batch_normalization_74/batchnorm/RsqrtRsqrt@feed_forward_sub_net_12/batch_normalization_74/batchnorm/add:z:0*
T0*
_output_shapes
:2@
>feed_forward_sub_net_12/batch_normalization_74/batchnorm/Rsqrt?
Kfeed_forward_sub_net_12/batch_normalization_74/batchnorm/mul/ReadVariableOpReadVariableOpTfeed_forward_sub_net_12_batch_normalization_74_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02M
Kfeed_forward_sub_net_12/batch_normalization_74/batchnorm/mul/ReadVariableOp?
<feed_forward_sub_net_12/batch_normalization_74/batchnorm/mulMulBfeed_forward_sub_net_12/batch_normalization_74/batchnorm/Rsqrt:y:0Sfeed_forward_sub_net_12/batch_normalization_74/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_12/batch_normalization_74/batchnorm/mul?
>feed_forward_sub_net_12/batch_normalization_74/batchnorm/mul_1Mul1feed_forward_sub_net_12/dense_61/MatMul:product:0@feed_forward_sub_net_12/batch_normalization_74/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2@
>feed_forward_sub_net_12/batch_normalization_74/batchnorm/mul_1?
Ifeed_forward_sub_net_12/batch_normalization_74/batchnorm/ReadVariableOp_1ReadVariableOpRfeed_forward_sub_net_12_batch_normalization_74_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02K
Ifeed_forward_sub_net_12/batch_normalization_74/batchnorm/ReadVariableOp_1?
>feed_forward_sub_net_12/batch_normalization_74/batchnorm/mul_2MulQfeed_forward_sub_net_12/batch_normalization_74/batchnorm/ReadVariableOp_1:value:0@feed_forward_sub_net_12/batch_normalization_74/batchnorm/mul:z:0*
T0*
_output_shapes
:2@
>feed_forward_sub_net_12/batch_normalization_74/batchnorm/mul_2?
Ifeed_forward_sub_net_12/batch_normalization_74/batchnorm/ReadVariableOp_2ReadVariableOpRfeed_forward_sub_net_12_batch_normalization_74_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02K
Ifeed_forward_sub_net_12/batch_normalization_74/batchnorm/ReadVariableOp_2?
<feed_forward_sub_net_12/batch_normalization_74/batchnorm/subSubQfeed_forward_sub_net_12/batch_normalization_74/batchnorm/ReadVariableOp_2:value:0Bfeed_forward_sub_net_12/batch_normalization_74/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_12/batch_normalization_74/batchnorm/sub?
>feed_forward_sub_net_12/batch_normalization_74/batchnorm/add_1AddV2Bfeed_forward_sub_net_12/batch_normalization_74/batchnorm/mul_1:z:0@feed_forward_sub_net_12/batch_normalization_74/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2@
>feed_forward_sub_net_12/batch_normalization_74/batchnorm/add_1?
feed_forward_sub_net_12/Relu_1ReluBfeed_forward_sub_net_12/batch_normalization_74/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2 
feed_forward_sub_net_12/Relu_1?
6feed_forward_sub_net_12/dense_62/MatMul/ReadVariableOpReadVariableOp?feed_forward_sub_net_12_dense_62_matmul_readvariableop_resource*
_output_shapes

:*
dtype028
6feed_forward_sub_net_12/dense_62/MatMul/ReadVariableOp?
'feed_forward_sub_net_12/dense_62/MatMulMatMul,feed_forward_sub_net_12/Relu_1:activations:0>feed_forward_sub_net_12/dense_62/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2)
'feed_forward_sub_net_12/dense_62/MatMul?
Gfeed_forward_sub_net_12/batch_normalization_75/batchnorm/ReadVariableOpReadVariableOpPfeed_forward_sub_net_12_batch_normalization_75_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02I
Gfeed_forward_sub_net_12/batch_normalization_75/batchnorm/ReadVariableOp?
>feed_forward_sub_net_12/batch_normalization_75/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2@
>feed_forward_sub_net_12/batch_normalization_75/batchnorm/add/y?
<feed_forward_sub_net_12/batch_normalization_75/batchnorm/addAddV2Ofeed_forward_sub_net_12/batch_normalization_75/batchnorm/ReadVariableOp:value:0Gfeed_forward_sub_net_12/batch_normalization_75/batchnorm/add/y:output:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_12/batch_normalization_75/batchnorm/add?
>feed_forward_sub_net_12/batch_normalization_75/batchnorm/RsqrtRsqrt@feed_forward_sub_net_12/batch_normalization_75/batchnorm/add:z:0*
T0*
_output_shapes
:2@
>feed_forward_sub_net_12/batch_normalization_75/batchnorm/Rsqrt?
Kfeed_forward_sub_net_12/batch_normalization_75/batchnorm/mul/ReadVariableOpReadVariableOpTfeed_forward_sub_net_12_batch_normalization_75_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02M
Kfeed_forward_sub_net_12/batch_normalization_75/batchnorm/mul/ReadVariableOp?
<feed_forward_sub_net_12/batch_normalization_75/batchnorm/mulMulBfeed_forward_sub_net_12/batch_normalization_75/batchnorm/Rsqrt:y:0Sfeed_forward_sub_net_12/batch_normalization_75/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_12/batch_normalization_75/batchnorm/mul?
>feed_forward_sub_net_12/batch_normalization_75/batchnorm/mul_1Mul1feed_forward_sub_net_12/dense_62/MatMul:product:0@feed_forward_sub_net_12/batch_normalization_75/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2@
>feed_forward_sub_net_12/batch_normalization_75/batchnorm/mul_1?
Ifeed_forward_sub_net_12/batch_normalization_75/batchnorm/ReadVariableOp_1ReadVariableOpRfeed_forward_sub_net_12_batch_normalization_75_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02K
Ifeed_forward_sub_net_12/batch_normalization_75/batchnorm/ReadVariableOp_1?
>feed_forward_sub_net_12/batch_normalization_75/batchnorm/mul_2MulQfeed_forward_sub_net_12/batch_normalization_75/batchnorm/ReadVariableOp_1:value:0@feed_forward_sub_net_12/batch_normalization_75/batchnorm/mul:z:0*
T0*
_output_shapes
:2@
>feed_forward_sub_net_12/batch_normalization_75/batchnorm/mul_2?
Ifeed_forward_sub_net_12/batch_normalization_75/batchnorm/ReadVariableOp_2ReadVariableOpRfeed_forward_sub_net_12_batch_normalization_75_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02K
Ifeed_forward_sub_net_12/batch_normalization_75/batchnorm/ReadVariableOp_2?
<feed_forward_sub_net_12/batch_normalization_75/batchnorm/subSubQfeed_forward_sub_net_12/batch_normalization_75/batchnorm/ReadVariableOp_2:value:0Bfeed_forward_sub_net_12/batch_normalization_75/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_12/batch_normalization_75/batchnorm/sub?
>feed_forward_sub_net_12/batch_normalization_75/batchnorm/add_1AddV2Bfeed_forward_sub_net_12/batch_normalization_75/batchnorm/mul_1:z:0@feed_forward_sub_net_12/batch_normalization_75/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2@
>feed_forward_sub_net_12/batch_normalization_75/batchnorm/add_1?
feed_forward_sub_net_12/Relu_2ReluBfeed_forward_sub_net_12/batch_normalization_75/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2 
feed_forward_sub_net_12/Relu_2?
6feed_forward_sub_net_12/dense_63/MatMul/ReadVariableOpReadVariableOp?feed_forward_sub_net_12_dense_63_matmul_readvariableop_resource*
_output_shapes

:*
dtype028
6feed_forward_sub_net_12/dense_63/MatMul/ReadVariableOp?
'feed_forward_sub_net_12/dense_63/MatMulMatMul,feed_forward_sub_net_12/Relu_2:activations:0>feed_forward_sub_net_12/dense_63/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2)
'feed_forward_sub_net_12/dense_63/MatMul?
Gfeed_forward_sub_net_12/batch_normalization_76/batchnorm/ReadVariableOpReadVariableOpPfeed_forward_sub_net_12_batch_normalization_76_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02I
Gfeed_forward_sub_net_12/batch_normalization_76/batchnorm/ReadVariableOp?
>feed_forward_sub_net_12/batch_normalization_76/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2@
>feed_forward_sub_net_12/batch_normalization_76/batchnorm/add/y?
<feed_forward_sub_net_12/batch_normalization_76/batchnorm/addAddV2Ofeed_forward_sub_net_12/batch_normalization_76/batchnorm/ReadVariableOp:value:0Gfeed_forward_sub_net_12/batch_normalization_76/batchnorm/add/y:output:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_12/batch_normalization_76/batchnorm/add?
>feed_forward_sub_net_12/batch_normalization_76/batchnorm/RsqrtRsqrt@feed_forward_sub_net_12/batch_normalization_76/batchnorm/add:z:0*
T0*
_output_shapes
:2@
>feed_forward_sub_net_12/batch_normalization_76/batchnorm/Rsqrt?
Kfeed_forward_sub_net_12/batch_normalization_76/batchnorm/mul/ReadVariableOpReadVariableOpTfeed_forward_sub_net_12_batch_normalization_76_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02M
Kfeed_forward_sub_net_12/batch_normalization_76/batchnorm/mul/ReadVariableOp?
<feed_forward_sub_net_12/batch_normalization_76/batchnorm/mulMulBfeed_forward_sub_net_12/batch_normalization_76/batchnorm/Rsqrt:y:0Sfeed_forward_sub_net_12/batch_normalization_76/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_12/batch_normalization_76/batchnorm/mul?
>feed_forward_sub_net_12/batch_normalization_76/batchnorm/mul_1Mul1feed_forward_sub_net_12/dense_63/MatMul:product:0@feed_forward_sub_net_12/batch_normalization_76/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2@
>feed_forward_sub_net_12/batch_normalization_76/batchnorm/mul_1?
Ifeed_forward_sub_net_12/batch_normalization_76/batchnorm/ReadVariableOp_1ReadVariableOpRfeed_forward_sub_net_12_batch_normalization_76_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02K
Ifeed_forward_sub_net_12/batch_normalization_76/batchnorm/ReadVariableOp_1?
>feed_forward_sub_net_12/batch_normalization_76/batchnorm/mul_2MulQfeed_forward_sub_net_12/batch_normalization_76/batchnorm/ReadVariableOp_1:value:0@feed_forward_sub_net_12/batch_normalization_76/batchnorm/mul:z:0*
T0*
_output_shapes
:2@
>feed_forward_sub_net_12/batch_normalization_76/batchnorm/mul_2?
Ifeed_forward_sub_net_12/batch_normalization_76/batchnorm/ReadVariableOp_2ReadVariableOpRfeed_forward_sub_net_12_batch_normalization_76_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02K
Ifeed_forward_sub_net_12/batch_normalization_76/batchnorm/ReadVariableOp_2?
<feed_forward_sub_net_12/batch_normalization_76/batchnorm/subSubQfeed_forward_sub_net_12/batch_normalization_76/batchnorm/ReadVariableOp_2:value:0Bfeed_forward_sub_net_12/batch_normalization_76/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_12/batch_normalization_76/batchnorm/sub?
>feed_forward_sub_net_12/batch_normalization_76/batchnorm/add_1AddV2Bfeed_forward_sub_net_12/batch_normalization_76/batchnorm/mul_1:z:0@feed_forward_sub_net_12/batch_normalization_76/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2@
>feed_forward_sub_net_12/batch_normalization_76/batchnorm/add_1?
feed_forward_sub_net_12/Relu_3ReluBfeed_forward_sub_net_12/batch_normalization_76/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2 
feed_forward_sub_net_12/Relu_3?
6feed_forward_sub_net_12/dense_64/MatMul/ReadVariableOpReadVariableOp?feed_forward_sub_net_12_dense_64_matmul_readvariableop_resource*
_output_shapes

:
*
dtype028
6feed_forward_sub_net_12/dense_64/MatMul/ReadVariableOp?
'feed_forward_sub_net_12/dense_64/MatMulMatMul,feed_forward_sub_net_12/Relu_3:activations:0>feed_forward_sub_net_12/dense_64/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2)
'feed_forward_sub_net_12/dense_64/MatMul?
7feed_forward_sub_net_12/dense_64/BiasAdd/ReadVariableOpReadVariableOp@feed_forward_sub_net_12_dense_64_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype029
7feed_forward_sub_net_12/dense_64/BiasAdd/ReadVariableOp?
(feed_forward_sub_net_12/dense_64/BiasAddBiasAdd1feed_forward_sub_net_12/dense_64/MatMul:product:0?feed_forward_sub_net_12/dense_64/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2*
(feed_forward_sub_net_12/dense_64/BiasAdd?
IdentityIdentity1feed_forward_sub_net_12/dense_64/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOpH^feed_forward_sub_net_12/batch_normalization_72/batchnorm/ReadVariableOpJ^feed_forward_sub_net_12/batch_normalization_72/batchnorm/ReadVariableOp_1J^feed_forward_sub_net_12/batch_normalization_72/batchnorm/ReadVariableOp_2L^feed_forward_sub_net_12/batch_normalization_72/batchnorm/mul/ReadVariableOpH^feed_forward_sub_net_12/batch_normalization_73/batchnorm/ReadVariableOpJ^feed_forward_sub_net_12/batch_normalization_73/batchnorm/ReadVariableOp_1J^feed_forward_sub_net_12/batch_normalization_73/batchnorm/ReadVariableOp_2L^feed_forward_sub_net_12/batch_normalization_73/batchnorm/mul/ReadVariableOpH^feed_forward_sub_net_12/batch_normalization_74/batchnorm/ReadVariableOpJ^feed_forward_sub_net_12/batch_normalization_74/batchnorm/ReadVariableOp_1J^feed_forward_sub_net_12/batch_normalization_74/batchnorm/ReadVariableOp_2L^feed_forward_sub_net_12/batch_normalization_74/batchnorm/mul/ReadVariableOpH^feed_forward_sub_net_12/batch_normalization_75/batchnorm/ReadVariableOpJ^feed_forward_sub_net_12/batch_normalization_75/batchnorm/ReadVariableOp_1J^feed_forward_sub_net_12/batch_normalization_75/batchnorm/ReadVariableOp_2L^feed_forward_sub_net_12/batch_normalization_75/batchnorm/mul/ReadVariableOpH^feed_forward_sub_net_12/batch_normalization_76/batchnorm/ReadVariableOpJ^feed_forward_sub_net_12/batch_normalization_76/batchnorm/ReadVariableOp_1J^feed_forward_sub_net_12/batch_normalization_76/batchnorm/ReadVariableOp_2L^feed_forward_sub_net_12/batch_normalization_76/batchnorm/mul/ReadVariableOp7^feed_forward_sub_net_12/dense_60/MatMul/ReadVariableOp7^feed_forward_sub_net_12/dense_61/MatMul/ReadVariableOp7^feed_forward_sub_net_12/dense_62/MatMul/ReadVariableOp7^feed_forward_sub_net_12/dense_63/MatMul/ReadVariableOp8^feed_forward_sub_net_12/dense_64/BiasAdd/ReadVariableOp7^feed_forward_sub_net_12/dense_64/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????
: : : : : : : : : : : : : : : : : : : : : : : : : : 2?
Gfeed_forward_sub_net_12/batch_normalization_72/batchnorm/ReadVariableOpGfeed_forward_sub_net_12/batch_normalization_72/batchnorm/ReadVariableOp2?
Ifeed_forward_sub_net_12/batch_normalization_72/batchnorm/ReadVariableOp_1Ifeed_forward_sub_net_12/batch_normalization_72/batchnorm/ReadVariableOp_12?
Ifeed_forward_sub_net_12/batch_normalization_72/batchnorm/ReadVariableOp_2Ifeed_forward_sub_net_12/batch_normalization_72/batchnorm/ReadVariableOp_22?
Kfeed_forward_sub_net_12/batch_normalization_72/batchnorm/mul/ReadVariableOpKfeed_forward_sub_net_12/batch_normalization_72/batchnorm/mul/ReadVariableOp2?
Gfeed_forward_sub_net_12/batch_normalization_73/batchnorm/ReadVariableOpGfeed_forward_sub_net_12/batch_normalization_73/batchnorm/ReadVariableOp2?
Ifeed_forward_sub_net_12/batch_normalization_73/batchnorm/ReadVariableOp_1Ifeed_forward_sub_net_12/batch_normalization_73/batchnorm/ReadVariableOp_12?
Ifeed_forward_sub_net_12/batch_normalization_73/batchnorm/ReadVariableOp_2Ifeed_forward_sub_net_12/batch_normalization_73/batchnorm/ReadVariableOp_22?
Kfeed_forward_sub_net_12/batch_normalization_73/batchnorm/mul/ReadVariableOpKfeed_forward_sub_net_12/batch_normalization_73/batchnorm/mul/ReadVariableOp2?
Gfeed_forward_sub_net_12/batch_normalization_74/batchnorm/ReadVariableOpGfeed_forward_sub_net_12/batch_normalization_74/batchnorm/ReadVariableOp2?
Ifeed_forward_sub_net_12/batch_normalization_74/batchnorm/ReadVariableOp_1Ifeed_forward_sub_net_12/batch_normalization_74/batchnorm/ReadVariableOp_12?
Ifeed_forward_sub_net_12/batch_normalization_74/batchnorm/ReadVariableOp_2Ifeed_forward_sub_net_12/batch_normalization_74/batchnorm/ReadVariableOp_22?
Kfeed_forward_sub_net_12/batch_normalization_74/batchnorm/mul/ReadVariableOpKfeed_forward_sub_net_12/batch_normalization_74/batchnorm/mul/ReadVariableOp2?
Gfeed_forward_sub_net_12/batch_normalization_75/batchnorm/ReadVariableOpGfeed_forward_sub_net_12/batch_normalization_75/batchnorm/ReadVariableOp2?
Ifeed_forward_sub_net_12/batch_normalization_75/batchnorm/ReadVariableOp_1Ifeed_forward_sub_net_12/batch_normalization_75/batchnorm/ReadVariableOp_12?
Ifeed_forward_sub_net_12/batch_normalization_75/batchnorm/ReadVariableOp_2Ifeed_forward_sub_net_12/batch_normalization_75/batchnorm/ReadVariableOp_22?
Kfeed_forward_sub_net_12/batch_normalization_75/batchnorm/mul/ReadVariableOpKfeed_forward_sub_net_12/batch_normalization_75/batchnorm/mul/ReadVariableOp2?
Gfeed_forward_sub_net_12/batch_normalization_76/batchnorm/ReadVariableOpGfeed_forward_sub_net_12/batch_normalization_76/batchnorm/ReadVariableOp2?
Ifeed_forward_sub_net_12/batch_normalization_76/batchnorm/ReadVariableOp_1Ifeed_forward_sub_net_12/batch_normalization_76/batchnorm/ReadVariableOp_12?
Ifeed_forward_sub_net_12/batch_normalization_76/batchnorm/ReadVariableOp_2Ifeed_forward_sub_net_12/batch_normalization_76/batchnorm/ReadVariableOp_22?
Kfeed_forward_sub_net_12/batch_normalization_76/batchnorm/mul/ReadVariableOpKfeed_forward_sub_net_12/batch_normalization_76/batchnorm/mul/ReadVariableOp2p
6feed_forward_sub_net_12/dense_60/MatMul/ReadVariableOp6feed_forward_sub_net_12/dense_60/MatMul/ReadVariableOp2p
6feed_forward_sub_net_12/dense_61/MatMul/ReadVariableOp6feed_forward_sub_net_12/dense_61/MatMul/ReadVariableOp2p
6feed_forward_sub_net_12/dense_62/MatMul/ReadVariableOp6feed_forward_sub_net_12/dense_62/MatMul/ReadVariableOp2p
6feed_forward_sub_net_12/dense_63/MatMul/ReadVariableOp6feed_forward_sub_net_12/dense_63/MatMul/ReadVariableOp2r
7feed_forward_sub_net_12/dense_64/BiasAdd/ReadVariableOp7feed_forward_sub_net_12/dense_64/BiasAdd/ReadVariableOp2p
6feed_forward_sub_net_12/dense_64/MatMul/ReadVariableOp6feed_forward_sub_net_12/dense_64/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????

!
_user_specified_name	input_1
??
?
T__inference_feed_forward_sub_net_12_layer_call_and_return_conditional_losses_7056066
xL
>batch_normalization_72_assignmovingavg_readvariableop_resource:
N
@batch_normalization_72_assignmovingavg_1_readvariableop_resource:
J
<batch_normalization_72_batchnorm_mul_readvariableop_resource:
F
8batch_normalization_72_batchnorm_readvariableop_resource:
9
'dense_60_matmul_readvariableop_resource:
L
>batch_normalization_73_assignmovingavg_readvariableop_resource:N
@batch_normalization_73_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_73_batchnorm_mul_readvariableop_resource:F
8batch_normalization_73_batchnorm_readvariableop_resource:9
'dense_61_matmul_readvariableop_resource:L
>batch_normalization_74_assignmovingavg_readvariableop_resource:N
@batch_normalization_74_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_74_batchnorm_mul_readvariableop_resource:F
8batch_normalization_74_batchnorm_readvariableop_resource:9
'dense_62_matmul_readvariableop_resource:L
>batch_normalization_75_assignmovingavg_readvariableop_resource:N
@batch_normalization_75_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_75_batchnorm_mul_readvariableop_resource:F
8batch_normalization_75_batchnorm_readvariableop_resource:9
'dense_63_matmul_readvariableop_resource:L
>batch_normalization_76_assignmovingavg_readvariableop_resource:N
@batch_normalization_76_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_76_batchnorm_mul_readvariableop_resource:F
8batch_normalization_76_batchnorm_readvariableop_resource:9
'dense_64_matmul_readvariableop_resource:
6
(dense_64_biasadd_readvariableop_resource:

identity??&batch_normalization_72/AssignMovingAvg?5batch_normalization_72/AssignMovingAvg/ReadVariableOp?(batch_normalization_72/AssignMovingAvg_1?7batch_normalization_72/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_72/batchnorm/ReadVariableOp?3batch_normalization_72/batchnorm/mul/ReadVariableOp?&batch_normalization_73/AssignMovingAvg?5batch_normalization_73/AssignMovingAvg/ReadVariableOp?(batch_normalization_73/AssignMovingAvg_1?7batch_normalization_73/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_73/batchnorm/ReadVariableOp?3batch_normalization_73/batchnorm/mul/ReadVariableOp?&batch_normalization_74/AssignMovingAvg?5batch_normalization_74/AssignMovingAvg/ReadVariableOp?(batch_normalization_74/AssignMovingAvg_1?7batch_normalization_74/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_74/batchnorm/ReadVariableOp?3batch_normalization_74/batchnorm/mul/ReadVariableOp?&batch_normalization_75/AssignMovingAvg?5batch_normalization_75/AssignMovingAvg/ReadVariableOp?(batch_normalization_75/AssignMovingAvg_1?7batch_normalization_75/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_75/batchnorm/ReadVariableOp?3batch_normalization_75/batchnorm/mul/ReadVariableOp?&batch_normalization_76/AssignMovingAvg?5batch_normalization_76/AssignMovingAvg/ReadVariableOp?(batch_normalization_76/AssignMovingAvg_1?7batch_normalization_76/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_76/batchnorm/ReadVariableOp?3batch_normalization_76/batchnorm/mul/ReadVariableOp?dense_60/MatMul/ReadVariableOp?dense_61/MatMul/ReadVariableOp?dense_62/MatMul/ReadVariableOp?dense_63/MatMul/ReadVariableOp?dense_64/BiasAdd/ReadVariableOp?dense_64/MatMul/ReadVariableOp?
5batch_normalization_72/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_72/moments/mean/reduction_indices?
#batch_normalization_72/moments/meanMeanx>batch_normalization_72/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2%
#batch_normalization_72/moments/mean?
+batch_normalization_72/moments/StopGradientStopGradient,batch_normalization_72/moments/mean:output:0*
T0*
_output_shapes

:
2-
+batch_normalization_72/moments/StopGradient?
0batch_normalization_72/moments/SquaredDifferenceSquaredDifferencex4batch_normalization_72/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????
22
0batch_normalization_72/moments/SquaredDifference?
9batch_normalization_72/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_72/moments/variance/reduction_indices?
'batch_normalization_72/moments/varianceMean4batch_normalization_72/moments/SquaredDifference:z:0Bbatch_normalization_72/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2)
'batch_normalization_72/moments/variance?
&batch_normalization_72/moments/SqueezeSqueeze,batch_normalization_72/moments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2(
&batch_normalization_72/moments/Squeeze?
(batch_normalization_72/moments/Squeeze_1Squeeze0batch_normalization_72/moments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2*
(batch_normalization_72/moments/Squeeze_1?
,batch_normalization_72/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_72/AssignMovingAvg/decay?
+batch_normalization_72/AssignMovingAvg/CastCast5batch_normalization_72/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_72/AssignMovingAvg/Cast?
5batch_normalization_72/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_72_assignmovingavg_readvariableop_resource*
_output_shapes
:
*
dtype027
5batch_normalization_72/AssignMovingAvg/ReadVariableOp?
*batch_normalization_72/AssignMovingAvg/subSub=batch_normalization_72/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_72/moments/Squeeze:output:0*
T0*
_output_shapes
:
2,
*batch_normalization_72/AssignMovingAvg/sub?
*batch_normalization_72/AssignMovingAvg/mulMul.batch_normalization_72/AssignMovingAvg/sub:z:0/batch_normalization_72/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:
2,
*batch_normalization_72/AssignMovingAvg/mul?
&batch_normalization_72/AssignMovingAvgAssignSubVariableOp>batch_normalization_72_assignmovingavg_readvariableop_resource.batch_normalization_72/AssignMovingAvg/mul:z:06^batch_normalization_72/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_72/AssignMovingAvg?
.batch_normalization_72/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_72/AssignMovingAvg_1/decay?
-batch_normalization_72/AssignMovingAvg_1/CastCast7batch_normalization_72/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_72/AssignMovingAvg_1/Cast?
7batch_normalization_72/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_72_assignmovingavg_1_readvariableop_resource*
_output_shapes
:
*
dtype029
7batch_normalization_72/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_72/AssignMovingAvg_1/subSub?batch_normalization_72/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_72/moments/Squeeze_1:output:0*
T0*
_output_shapes
:
2.
,batch_normalization_72/AssignMovingAvg_1/sub?
,batch_normalization_72/AssignMovingAvg_1/mulMul0batch_normalization_72/AssignMovingAvg_1/sub:z:01batch_normalization_72/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:
2.
,batch_normalization_72/AssignMovingAvg_1/mul?
(batch_normalization_72/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_72_assignmovingavg_1_readvariableop_resource0batch_normalization_72/AssignMovingAvg_1/mul:z:08^batch_normalization_72/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_72/AssignMovingAvg_1?
&batch_normalization_72/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2(
&batch_normalization_72/batchnorm/add/y?
$batch_normalization_72/batchnorm/addAddV21batch_normalization_72/moments/Squeeze_1:output:0/batch_normalization_72/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2&
$batch_normalization_72/batchnorm/add?
&batch_normalization_72/batchnorm/RsqrtRsqrt(batch_normalization_72/batchnorm/add:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_72/batchnorm/Rsqrt?
3batch_normalization_72/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_72_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype025
3batch_normalization_72/batchnorm/mul/ReadVariableOp?
$batch_normalization_72/batchnorm/mulMul*batch_normalization_72/batchnorm/Rsqrt:y:0;batch_normalization_72/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2&
$batch_normalization_72/batchnorm/mul?
&batch_normalization_72/batchnorm/mul_1Mulx(batch_normalization_72/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????
2(
&batch_normalization_72/batchnorm/mul_1?
&batch_normalization_72/batchnorm/mul_2Mul/batch_normalization_72/moments/Squeeze:output:0(batch_normalization_72/batchnorm/mul:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_72/batchnorm/mul_2?
/batch_normalization_72/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_72_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype021
/batch_normalization_72/batchnorm/ReadVariableOp?
$batch_normalization_72/batchnorm/subSub7batch_normalization_72/batchnorm/ReadVariableOp:value:0*batch_normalization_72/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2&
$batch_normalization_72/batchnorm/sub?
&batch_normalization_72/batchnorm/add_1AddV2*batch_normalization_72/batchnorm/mul_1:z:0(batch_normalization_72/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????
2(
&batch_normalization_72/batchnorm/add_1?
dense_60/MatMul/ReadVariableOpReadVariableOp'dense_60_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_60/MatMul/ReadVariableOp?
dense_60/MatMulMatMul*batch_normalization_72/batchnorm/add_1:z:0&dense_60/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_60/MatMul?
5batch_normalization_73/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_73/moments/mean/reduction_indices?
#batch_normalization_73/moments/meanMeandense_60/MatMul:product:0>batch_normalization_73/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_73/moments/mean?
+batch_normalization_73/moments/StopGradientStopGradient,batch_normalization_73/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_73/moments/StopGradient?
0batch_normalization_73/moments/SquaredDifferenceSquaredDifferencedense_60/MatMul:product:04batch_normalization_73/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????22
0batch_normalization_73/moments/SquaredDifference?
9batch_normalization_73/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_73/moments/variance/reduction_indices?
'batch_normalization_73/moments/varianceMean4batch_normalization_73/moments/SquaredDifference:z:0Bbatch_normalization_73/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_73/moments/variance?
&batch_normalization_73/moments/SqueezeSqueeze,batch_normalization_73/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_73/moments/Squeeze?
(batch_normalization_73/moments/Squeeze_1Squeeze0batch_normalization_73/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_73/moments/Squeeze_1?
,batch_normalization_73/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_73/AssignMovingAvg/decay?
+batch_normalization_73/AssignMovingAvg/CastCast5batch_normalization_73/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_73/AssignMovingAvg/Cast?
5batch_normalization_73/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_73_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_73/AssignMovingAvg/ReadVariableOp?
*batch_normalization_73/AssignMovingAvg/subSub=batch_normalization_73/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_73/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_73/AssignMovingAvg/sub?
*batch_normalization_73/AssignMovingAvg/mulMul.batch_normalization_73/AssignMovingAvg/sub:z:0/batch_normalization_73/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_73/AssignMovingAvg/mul?
&batch_normalization_73/AssignMovingAvgAssignSubVariableOp>batch_normalization_73_assignmovingavg_readvariableop_resource.batch_normalization_73/AssignMovingAvg/mul:z:06^batch_normalization_73/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_73/AssignMovingAvg?
.batch_normalization_73/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_73/AssignMovingAvg_1/decay?
-batch_normalization_73/AssignMovingAvg_1/CastCast7batch_normalization_73/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_73/AssignMovingAvg_1/Cast?
7batch_normalization_73/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_73_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_73/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_73/AssignMovingAvg_1/subSub?batch_normalization_73/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_73/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_73/AssignMovingAvg_1/sub?
,batch_normalization_73/AssignMovingAvg_1/mulMul0batch_normalization_73/AssignMovingAvg_1/sub:z:01batch_normalization_73/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_73/AssignMovingAvg_1/mul?
(batch_normalization_73/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_73_assignmovingavg_1_readvariableop_resource0batch_normalization_73/AssignMovingAvg_1/mul:z:08^batch_normalization_73/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_73/AssignMovingAvg_1?
&batch_normalization_73/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2(
&batch_normalization_73/batchnorm/add/y?
$batch_normalization_73/batchnorm/addAddV21batch_normalization_73/moments/Squeeze_1:output:0/batch_normalization_73/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_73/batchnorm/add?
&batch_normalization_73/batchnorm/RsqrtRsqrt(batch_normalization_73/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_73/batchnorm/Rsqrt?
3batch_normalization_73/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_73_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_73/batchnorm/mul/ReadVariableOp?
$batch_normalization_73/batchnorm/mulMul*batch_normalization_73/batchnorm/Rsqrt:y:0;batch_normalization_73/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_73/batchnorm/mul?
&batch_normalization_73/batchnorm/mul_1Muldense_60/MatMul:product:0(batch_normalization_73/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_73/batchnorm/mul_1?
&batch_normalization_73/batchnorm/mul_2Mul/batch_normalization_73/moments/Squeeze:output:0(batch_normalization_73/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_73/batchnorm/mul_2?
/batch_normalization_73/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_73_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_73/batchnorm/ReadVariableOp?
$batch_normalization_73/batchnorm/subSub7batch_normalization_73/batchnorm/ReadVariableOp:value:0*batch_normalization_73/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_73/batchnorm/sub?
&batch_normalization_73/batchnorm/add_1AddV2*batch_normalization_73/batchnorm/mul_1:z:0(batch_normalization_73/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_73/batchnorm/add_1r
ReluRelu*batch_normalization_73/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu?
dense_61/MatMul/ReadVariableOpReadVariableOp'dense_61_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_61/MatMul/ReadVariableOp?
dense_61/MatMulMatMulRelu:activations:0&dense_61/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_61/MatMul?
5batch_normalization_74/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_74/moments/mean/reduction_indices?
#batch_normalization_74/moments/meanMeandense_61/MatMul:product:0>batch_normalization_74/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_74/moments/mean?
+batch_normalization_74/moments/StopGradientStopGradient,batch_normalization_74/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_74/moments/StopGradient?
0batch_normalization_74/moments/SquaredDifferenceSquaredDifferencedense_61/MatMul:product:04batch_normalization_74/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????22
0batch_normalization_74/moments/SquaredDifference?
9batch_normalization_74/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_74/moments/variance/reduction_indices?
'batch_normalization_74/moments/varianceMean4batch_normalization_74/moments/SquaredDifference:z:0Bbatch_normalization_74/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_74/moments/variance?
&batch_normalization_74/moments/SqueezeSqueeze,batch_normalization_74/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_74/moments/Squeeze?
(batch_normalization_74/moments/Squeeze_1Squeeze0batch_normalization_74/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_74/moments/Squeeze_1?
,batch_normalization_74/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_74/AssignMovingAvg/decay?
+batch_normalization_74/AssignMovingAvg/CastCast5batch_normalization_74/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_74/AssignMovingAvg/Cast?
5batch_normalization_74/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_74_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_74/AssignMovingAvg/ReadVariableOp?
*batch_normalization_74/AssignMovingAvg/subSub=batch_normalization_74/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_74/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_74/AssignMovingAvg/sub?
*batch_normalization_74/AssignMovingAvg/mulMul.batch_normalization_74/AssignMovingAvg/sub:z:0/batch_normalization_74/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_74/AssignMovingAvg/mul?
&batch_normalization_74/AssignMovingAvgAssignSubVariableOp>batch_normalization_74_assignmovingavg_readvariableop_resource.batch_normalization_74/AssignMovingAvg/mul:z:06^batch_normalization_74/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_74/AssignMovingAvg?
.batch_normalization_74/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_74/AssignMovingAvg_1/decay?
-batch_normalization_74/AssignMovingAvg_1/CastCast7batch_normalization_74/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_74/AssignMovingAvg_1/Cast?
7batch_normalization_74/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_74_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_74/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_74/AssignMovingAvg_1/subSub?batch_normalization_74/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_74/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_74/AssignMovingAvg_1/sub?
,batch_normalization_74/AssignMovingAvg_1/mulMul0batch_normalization_74/AssignMovingAvg_1/sub:z:01batch_normalization_74/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_74/AssignMovingAvg_1/mul?
(batch_normalization_74/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_74_assignmovingavg_1_readvariableop_resource0batch_normalization_74/AssignMovingAvg_1/mul:z:08^batch_normalization_74/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_74/AssignMovingAvg_1?
&batch_normalization_74/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2(
&batch_normalization_74/batchnorm/add/y?
$batch_normalization_74/batchnorm/addAddV21batch_normalization_74/moments/Squeeze_1:output:0/batch_normalization_74/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_74/batchnorm/add?
&batch_normalization_74/batchnorm/RsqrtRsqrt(batch_normalization_74/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_74/batchnorm/Rsqrt?
3batch_normalization_74/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_74_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_74/batchnorm/mul/ReadVariableOp?
$batch_normalization_74/batchnorm/mulMul*batch_normalization_74/batchnorm/Rsqrt:y:0;batch_normalization_74/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_74/batchnorm/mul?
&batch_normalization_74/batchnorm/mul_1Muldense_61/MatMul:product:0(batch_normalization_74/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_74/batchnorm/mul_1?
&batch_normalization_74/batchnorm/mul_2Mul/batch_normalization_74/moments/Squeeze:output:0(batch_normalization_74/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_74/batchnorm/mul_2?
/batch_normalization_74/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_74_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_74/batchnorm/ReadVariableOp?
$batch_normalization_74/batchnorm/subSub7batch_normalization_74/batchnorm/ReadVariableOp:value:0*batch_normalization_74/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_74/batchnorm/sub?
&batch_normalization_74/batchnorm/add_1AddV2*batch_normalization_74/batchnorm/mul_1:z:0(batch_normalization_74/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_74/batchnorm/add_1v
Relu_1Relu*batch_normalization_74/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu_1?
dense_62/MatMul/ReadVariableOpReadVariableOp'dense_62_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_62/MatMul/ReadVariableOp?
dense_62/MatMulMatMulRelu_1:activations:0&dense_62/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_62/MatMul?
5batch_normalization_75/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_75/moments/mean/reduction_indices?
#batch_normalization_75/moments/meanMeandense_62/MatMul:product:0>batch_normalization_75/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_75/moments/mean?
+batch_normalization_75/moments/StopGradientStopGradient,batch_normalization_75/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_75/moments/StopGradient?
0batch_normalization_75/moments/SquaredDifferenceSquaredDifferencedense_62/MatMul:product:04batch_normalization_75/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????22
0batch_normalization_75/moments/SquaredDifference?
9batch_normalization_75/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_75/moments/variance/reduction_indices?
'batch_normalization_75/moments/varianceMean4batch_normalization_75/moments/SquaredDifference:z:0Bbatch_normalization_75/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_75/moments/variance?
&batch_normalization_75/moments/SqueezeSqueeze,batch_normalization_75/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_75/moments/Squeeze?
(batch_normalization_75/moments/Squeeze_1Squeeze0batch_normalization_75/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_75/moments/Squeeze_1?
,batch_normalization_75/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_75/AssignMovingAvg/decay?
+batch_normalization_75/AssignMovingAvg/CastCast5batch_normalization_75/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_75/AssignMovingAvg/Cast?
5batch_normalization_75/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_75_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_75/AssignMovingAvg/ReadVariableOp?
*batch_normalization_75/AssignMovingAvg/subSub=batch_normalization_75/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_75/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_75/AssignMovingAvg/sub?
*batch_normalization_75/AssignMovingAvg/mulMul.batch_normalization_75/AssignMovingAvg/sub:z:0/batch_normalization_75/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_75/AssignMovingAvg/mul?
&batch_normalization_75/AssignMovingAvgAssignSubVariableOp>batch_normalization_75_assignmovingavg_readvariableop_resource.batch_normalization_75/AssignMovingAvg/mul:z:06^batch_normalization_75/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_75/AssignMovingAvg?
.batch_normalization_75/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_75/AssignMovingAvg_1/decay?
-batch_normalization_75/AssignMovingAvg_1/CastCast7batch_normalization_75/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_75/AssignMovingAvg_1/Cast?
7batch_normalization_75/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_75_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_75/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_75/AssignMovingAvg_1/subSub?batch_normalization_75/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_75/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_75/AssignMovingAvg_1/sub?
,batch_normalization_75/AssignMovingAvg_1/mulMul0batch_normalization_75/AssignMovingAvg_1/sub:z:01batch_normalization_75/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_75/AssignMovingAvg_1/mul?
(batch_normalization_75/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_75_assignmovingavg_1_readvariableop_resource0batch_normalization_75/AssignMovingAvg_1/mul:z:08^batch_normalization_75/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_75/AssignMovingAvg_1?
&batch_normalization_75/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2(
&batch_normalization_75/batchnorm/add/y?
$batch_normalization_75/batchnorm/addAddV21batch_normalization_75/moments/Squeeze_1:output:0/batch_normalization_75/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_75/batchnorm/add?
&batch_normalization_75/batchnorm/RsqrtRsqrt(batch_normalization_75/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_75/batchnorm/Rsqrt?
3batch_normalization_75/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_75_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_75/batchnorm/mul/ReadVariableOp?
$batch_normalization_75/batchnorm/mulMul*batch_normalization_75/batchnorm/Rsqrt:y:0;batch_normalization_75/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_75/batchnorm/mul?
&batch_normalization_75/batchnorm/mul_1Muldense_62/MatMul:product:0(batch_normalization_75/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_75/batchnorm/mul_1?
&batch_normalization_75/batchnorm/mul_2Mul/batch_normalization_75/moments/Squeeze:output:0(batch_normalization_75/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_75/batchnorm/mul_2?
/batch_normalization_75/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_75_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_75/batchnorm/ReadVariableOp?
$batch_normalization_75/batchnorm/subSub7batch_normalization_75/batchnorm/ReadVariableOp:value:0*batch_normalization_75/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_75/batchnorm/sub?
&batch_normalization_75/batchnorm/add_1AddV2*batch_normalization_75/batchnorm/mul_1:z:0(batch_normalization_75/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_75/batchnorm/add_1v
Relu_2Relu*batch_normalization_75/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu_2?
dense_63/MatMul/ReadVariableOpReadVariableOp'dense_63_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_63/MatMul/ReadVariableOp?
dense_63/MatMulMatMulRelu_2:activations:0&dense_63/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_63/MatMul?
5batch_normalization_76/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_76/moments/mean/reduction_indices?
#batch_normalization_76/moments/meanMeandense_63/MatMul:product:0>batch_normalization_76/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_76/moments/mean?
+batch_normalization_76/moments/StopGradientStopGradient,batch_normalization_76/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_76/moments/StopGradient?
0batch_normalization_76/moments/SquaredDifferenceSquaredDifferencedense_63/MatMul:product:04batch_normalization_76/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????22
0batch_normalization_76/moments/SquaredDifference?
9batch_normalization_76/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_76/moments/variance/reduction_indices?
'batch_normalization_76/moments/varianceMean4batch_normalization_76/moments/SquaredDifference:z:0Bbatch_normalization_76/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_76/moments/variance?
&batch_normalization_76/moments/SqueezeSqueeze,batch_normalization_76/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_76/moments/Squeeze?
(batch_normalization_76/moments/Squeeze_1Squeeze0batch_normalization_76/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_76/moments/Squeeze_1?
,batch_normalization_76/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_76/AssignMovingAvg/decay?
+batch_normalization_76/AssignMovingAvg/CastCast5batch_normalization_76/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_76/AssignMovingAvg/Cast?
5batch_normalization_76/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_76_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_76/AssignMovingAvg/ReadVariableOp?
*batch_normalization_76/AssignMovingAvg/subSub=batch_normalization_76/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_76/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_76/AssignMovingAvg/sub?
*batch_normalization_76/AssignMovingAvg/mulMul.batch_normalization_76/AssignMovingAvg/sub:z:0/batch_normalization_76/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_76/AssignMovingAvg/mul?
&batch_normalization_76/AssignMovingAvgAssignSubVariableOp>batch_normalization_76_assignmovingavg_readvariableop_resource.batch_normalization_76/AssignMovingAvg/mul:z:06^batch_normalization_76/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_76/AssignMovingAvg?
.batch_normalization_76/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_76/AssignMovingAvg_1/decay?
-batch_normalization_76/AssignMovingAvg_1/CastCast7batch_normalization_76/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_76/AssignMovingAvg_1/Cast?
7batch_normalization_76/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_76_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_76/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_76/AssignMovingAvg_1/subSub?batch_normalization_76/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_76/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_76/AssignMovingAvg_1/sub?
,batch_normalization_76/AssignMovingAvg_1/mulMul0batch_normalization_76/AssignMovingAvg_1/sub:z:01batch_normalization_76/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_76/AssignMovingAvg_1/mul?
(batch_normalization_76/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_76_assignmovingavg_1_readvariableop_resource0batch_normalization_76/AssignMovingAvg_1/mul:z:08^batch_normalization_76/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_76/AssignMovingAvg_1?
&batch_normalization_76/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2(
&batch_normalization_76/batchnorm/add/y?
$batch_normalization_76/batchnorm/addAddV21batch_normalization_76/moments/Squeeze_1:output:0/batch_normalization_76/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_76/batchnorm/add?
&batch_normalization_76/batchnorm/RsqrtRsqrt(batch_normalization_76/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_76/batchnorm/Rsqrt?
3batch_normalization_76/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_76_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_76/batchnorm/mul/ReadVariableOp?
$batch_normalization_76/batchnorm/mulMul*batch_normalization_76/batchnorm/Rsqrt:y:0;batch_normalization_76/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_76/batchnorm/mul?
&batch_normalization_76/batchnorm/mul_1Muldense_63/MatMul:product:0(batch_normalization_76/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_76/batchnorm/mul_1?
&batch_normalization_76/batchnorm/mul_2Mul/batch_normalization_76/moments/Squeeze:output:0(batch_normalization_76/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_76/batchnorm/mul_2?
/batch_normalization_76/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_76_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_76/batchnorm/ReadVariableOp?
$batch_normalization_76/batchnorm/subSub7batch_normalization_76/batchnorm/ReadVariableOp:value:0*batch_normalization_76/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_76/batchnorm/sub?
&batch_normalization_76/batchnorm/add_1AddV2*batch_normalization_76/batchnorm/mul_1:z:0(batch_normalization_76/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_76/batchnorm/add_1v
Relu_3Relu*batch_normalization_76/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu_3?
dense_64/MatMul/ReadVariableOpReadVariableOp'dense_64_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_64/MatMul/ReadVariableOp?
dense_64/MatMulMatMulRelu_3:activations:0&dense_64/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_64/MatMul?
dense_64/BiasAdd/ReadVariableOpReadVariableOp(dense_64_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_64/BiasAdd/ReadVariableOp?
dense_64/BiasAddBiasAdddense_64/MatMul:product:0'dense_64/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_64/BiasAddt
IdentityIdentitydense_64/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp'^batch_normalization_72/AssignMovingAvg6^batch_normalization_72/AssignMovingAvg/ReadVariableOp)^batch_normalization_72/AssignMovingAvg_18^batch_normalization_72/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_72/batchnorm/ReadVariableOp4^batch_normalization_72/batchnorm/mul/ReadVariableOp'^batch_normalization_73/AssignMovingAvg6^batch_normalization_73/AssignMovingAvg/ReadVariableOp)^batch_normalization_73/AssignMovingAvg_18^batch_normalization_73/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_73/batchnorm/ReadVariableOp4^batch_normalization_73/batchnorm/mul/ReadVariableOp'^batch_normalization_74/AssignMovingAvg6^batch_normalization_74/AssignMovingAvg/ReadVariableOp)^batch_normalization_74/AssignMovingAvg_18^batch_normalization_74/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_74/batchnorm/ReadVariableOp4^batch_normalization_74/batchnorm/mul/ReadVariableOp'^batch_normalization_75/AssignMovingAvg6^batch_normalization_75/AssignMovingAvg/ReadVariableOp)^batch_normalization_75/AssignMovingAvg_18^batch_normalization_75/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_75/batchnorm/ReadVariableOp4^batch_normalization_75/batchnorm/mul/ReadVariableOp'^batch_normalization_76/AssignMovingAvg6^batch_normalization_76/AssignMovingAvg/ReadVariableOp)^batch_normalization_76/AssignMovingAvg_18^batch_normalization_76/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_76/batchnorm/ReadVariableOp4^batch_normalization_76/batchnorm/mul/ReadVariableOp^dense_60/MatMul/ReadVariableOp^dense_61/MatMul/ReadVariableOp^dense_62/MatMul/ReadVariableOp^dense_63/MatMul/ReadVariableOp ^dense_64/BiasAdd/ReadVariableOp^dense_64/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????
: : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_72/AssignMovingAvg&batch_normalization_72/AssignMovingAvg2n
5batch_normalization_72/AssignMovingAvg/ReadVariableOp5batch_normalization_72/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_72/AssignMovingAvg_1(batch_normalization_72/AssignMovingAvg_12r
7batch_normalization_72/AssignMovingAvg_1/ReadVariableOp7batch_normalization_72/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_72/batchnorm/ReadVariableOp/batch_normalization_72/batchnorm/ReadVariableOp2j
3batch_normalization_72/batchnorm/mul/ReadVariableOp3batch_normalization_72/batchnorm/mul/ReadVariableOp2P
&batch_normalization_73/AssignMovingAvg&batch_normalization_73/AssignMovingAvg2n
5batch_normalization_73/AssignMovingAvg/ReadVariableOp5batch_normalization_73/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_73/AssignMovingAvg_1(batch_normalization_73/AssignMovingAvg_12r
7batch_normalization_73/AssignMovingAvg_1/ReadVariableOp7batch_normalization_73/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_73/batchnorm/ReadVariableOp/batch_normalization_73/batchnorm/ReadVariableOp2j
3batch_normalization_73/batchnorm/mul/ReadVariableOp3batch_normalization_73/batchnorm/mul/ReadVariableOp2P
&batch_normalization_74/AssignMovingAvg&batch_normalization_74/AssignMovingAvg2n
5batch_normalization_74/AssignMovingAvg/ReadVariableOp5batch_normalization_74/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_74/AssignMovingAvg_1(batch_normalization_74/AssignMovingAvg_12r
7batch_normalization_74/AssignMovingAvg_1/ReadVariableOp7batch_normalization_74/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_74/batchnorm/ReadVariableOp/batch_normalization_74/batchnorm/ReadVariableOp2j
3batch_normalization_74/batchnorm/mul/ReadVariableOp3batch_normalization_74/batchnorm/mul/ReadVariableOp2P
&batch_normalization_75/AssignMovingAvg&batch_normalization_75/AssignMovingAvg2n
5batch_normalization_75/AssignMovingAvg/ReadVariableOp5batch_normalization_75/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_75/AssignMovingAvg_1(batch_normalization_75/AssignMovingAvg_12r
7batch_normalization_75/AssignMovingAvg_1/ReadVariableOp7batch_normalization_75/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_75/batchnorm/ReadVariableOp/batch_normalization_75/batchnorm/ReadVariableOp2j
3batch_normalization_75/batchnorm/mul/ReadVariableOp3batch_normalization_75/batchnorm/mul/ReadVariableOp2P
&batch_normalization_76/AssignMovingAvg&batch_normalization_76/AssignMovingAvg2n
5batch_normalization_76/AssignMovingAvg/ReadVariableOp5batch_normalization_76/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_76/AssignMovingAvg_1(batch_normalization_76/AssignMovingAvg_12r
7batch_normalization_76/AssignMovingAvg_1/ReadVariableOp7batch_normalization_76/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_76/batchnorm/ReadVariableOp/batch_normalization_76/batchnorm/ReadVariableOp2j
3batch_normalization_76/batchnorm/mul/ReadVariableOp3batch_normalization_76/batchnorm/mul/ReadVariableOp2@
dense_60/MatMul/ReadVariableOpdense_60/MatMul/ReadVariableOp2@
dense_61/MatMul/ReadVariableOpdense_61/MatMul/ReadVariableOp2@
dense_62/MatMul/ReadVariableOpdense_62/MatMul/ReadVariableOp2@
dense_63/MatMul/ReadVariableOpdense_63/MatMul/ReadVariableOp2B
dense_64/BiasAdd/ReadVariableOpdense_64/BiasAdd/ReadVariableOp2@
dense_64/MatMul/ReadVariableOpdense_64/MatMul/ReadVariableOp:J F
'
_output_shapes
:?????????


_user_specified_namex
?
?
S__inference_batch_normalization_76_layer_call_and_return_conditional_losses_7056732

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
 __inference__traced_save_7056944
file_prefixe
asavev2_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_72_gamma_read_readvariableopd
`savev2_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_72_beta_read_readvariableope
asavev2_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_73_gamma_read_readvariableopd
`savev2_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_73_beta_read_readvariableope
asavev2_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_74_gamma_read_readvariableopd
`savev2_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_74_beta_read_readvariableope
asavev2_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_75_gamma_read_readvariableopd
`savev2_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_75_beta_read_readvariableope
asavev2_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_76_gamma_read_readvariableopd
`savev2_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_76_beta_read_readvariableopk
gsavev2_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_72_moving_mean_read_readvariableopo
ksavev2_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_72_moving_variance_read_readvariableopk
gsavev2_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_73_moving_mean_read_readvariableopo
ksavev2_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_73_moving_variance_read_readvariableopk
gsavev2_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_74_moving_mean_read_readvariableopo
ksavev2_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_74_moving_variance_read_readvariableopk
gsavev2_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_75_moving_mean_read_readvariableopo
ksavev2_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_75_moving_variance_read_readvariableopk
gsavev2_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_76_moving_mean_read_readvariableopo
ksavev2_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_76_moving_variance_read_readvariableopX
Tsavev2_nonshared_model_1_feed_forward_sub_net_12_dense_60_kernel_read_readvariableopX
Tsavev2_nonshared_model_1_feed_forward_sub_net_12_dense_61_kernel_read_readvariableopX
Tsavev2_nonshared_model_1_feed_forward_sub_net_12_dense_62_kernel_read_readvariableopX
Tsavev2_nonshared_model_1_feed_forward_sub_net_12_dense_63_kernel_read_readvariableopX
Tsavev2_nonshared_model_1_feed_forward_sub_net_12_dense_64_kernel_read_readvariableopV
Rsavev2_nonshared_model_1_feed_forward_sub_net_12_dense_64_bias_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0asavev2_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_72_gamma_read_readvariableop`savev2_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_72_beta_read_readvariableopasavev2_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_73_gamma_read_readvariableop`savev2_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_73_beta_read_readvariableopasavev2_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_74_gamma_read_readvariableop`savev2_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_74_beta_read_readvariableopasavev2_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_75_gamma_read_readvariableop`savev2_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_75_beta_read_readvariableopasavev2_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_76_gamma_read_readvariableop`savev2_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_76_beta_read_readvariableopgsavev2_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_72_moving_mean_read_readvariableopksavev2_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_72_moving_variance_read_readvariableopgsavev2_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_73_moving_mean_read_readvariableopksavev2_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_73_moving_variance_read_readvariableopgsavev2_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_74_moving_mean_read_readvariableopksavev2_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_74_moving_variance_read_readvariableopgsavev2_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_75_moving_mean_read_readvariableopksavev2_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_75_moving_variance_read_readvariableopgsavev2_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_76_moving_mean_read_readvariableopksavev2_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_76_moving_variance_read_readvariableopTsavev2_nonshared_model_1_feed_forward_sub_net_12_dense_60_kernel_read_readvariableopTsavev2_nonshared_model_1_feed_forward_sub_net_12_dense_61_kernel_read_readvariableopTsavev2_nonshared_model_1_feed_forward_sub_net_12_dense_62_kernel_read_readvariableopTsavev2_nonshared_model_1_feed_forward_sub_net_12_dense_63_kernel_read_readvariableopTsavev2_nonshared_model_1_feed_forward_sub_net_12_dense_64_kernel_read_readvariableopRsavev2_nonshared_model_1_feed_forward_sub_net_12_dense_64_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
?E
?
T__inference_feed_forward_sub_net_12_layer_call_and_return_conditional_losses_7055009
x,
batch_normalization_72_7054899:
,
batch_normalization_72_7054901:
,
batch_normalization_72_7054903:
,
batch_normalization_72_7054905:
"
dense_60_7054916:
,
batch_normalization_73_7054919:,
batch_normalization_73_7054921:,
batch_normalization_73_7054923:,
batch_normalization_73_7054925:"
dense_61_7054937:,
batch_normalization_74_7054940:,
batch_normalization_74_7054942:,
batch_normalization_74_7054944:,
batch_normalization_74_7054946:"
dense_62_7054958:,
batch_normalization_75_7054961:,
batch_normalization_75_7054963:,
batch_normalization_75_7054965:,
batch_normalization_75_7054967:"
dense_63_7054979:,
batch_normalization_76_7054982:,
batch_normalization_76_7054984:,
batch_normalization_76_7054986:,
batch_normalization_76_7054988:"
dense_64_7055003:

dense_64_7055005:

identity??.batch_normalization_72/StatefulPartitionedCall?.batch_normalization_73/StatefulPartitionedCall?.batch_normalization_74/StatefulPartitionedCall?.batch_normalization_75/StatefulPartitionedCall?.batch_normalization_76/StatefulPartitionedCall? dense_60/StatefulPartitionedCall? dense_61/StatefulPartitionedCall? dense_62/StatefulPartitionedCall? dense_63/StatefulPartitionedCall? dense_64/StatefulPartitionedCall?
.batch_normalization_72/StatefulPartitionedCallStatefulPartitionedCallxbatch_normalization_72_7054899batch_normalization_72_7054901batch_normalization_72_7054903batch_normalization_72_7054905*
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
S__inference_batch_normalization_72_layer_call_and_return_conditional_losses_705408620
.batch_normalization_72/StatefulPartitionedCall?
 dense_60/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_72/StatefulPartitionedCall:output:0dense_60_7054916*
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
E__inference_dense_60_layer_call_and_return_conditional_losses_70549152"
 dense_60/StatefulPartitionedCall?
.batch_normalization_73/StatefulPartitionedCallStatefulPartitionedCall)dense_60/StatefulPartitionedCall:output:0batch_normalization_73_7054919batch_normalization_73_7054921batch_normalization_73_7054923batch_normalization_73_7054925*
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
S__inference_batch_normalization_73_layer_call_and_return_conditional_losses_705425220
.batch_normalization_73/StatefulPartitionedCall
ReluRelu7batch_normalization_73/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
Relu?
 dense_61/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0dense_61_7054937*
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
E__inference_dense_61_layer_call_and_return_conditional_losses_70549362"
 dense_61/StatefulPartitionedCall?
.batch_normalization_74/StatefulPartitionedCallStatefulPartitionedCall)dense_61/StatefulPartitionedCall:output:0batch_normalization_74_7054940batch_normalization_74_7054942batch_normalization_74_7054944batch_normalization_74_7054946*
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
S__inference_batch_normalization_74_layer_call_and_return_conditional_losses_705441820
.batch_normalization_74/StatefulPartitionedCall?
Relu_1Relu7batch_normalization_74/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
Relu_1?
 dense_62/StatefulPartitionedCallStatefulPartitionedCallRelu_1:activations:0dense_62_7054958*
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
E__inference_dense_62_layer_call_and_return_conditional_losses_70549572"
 dense_62/StatefulPartitionedCall?
.batch_normalization_75/StatefulPartitionedCallStatefulPartitionedCall)dense_62/StatefulPartitionedCall:output:0batch_normalization_75_7054961batch_normalization_75_7054963batch_normalization_75_7054965batch_normalization_75_7054967*
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
S__inference_batch_normalization_75_layer_call_and_return_conditional_losses_705458420
.batch_normalization_75/StatefulPartitionedCall?
Relu_2Relu7batch_normalization_75/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
Relu_2?
 dense_63/StatefulPartitionedCallStatefulPartitionedCallRelu_2:activations:0dense_63_7054979*
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
E__inference_dense_63_layer_call_and_return_conditional_losses_70549782"
 dense_63/StatefulPartitionedCall?
.batch_normalization_76/StatefulPartitionedCallStatefulPartitionedCall)dense_63/StatefulPartitionedCall:output:0batch_normalization_76_7054982batch_normalization_76_7054984batch_normalization_76_7054986batch_normalization_76_7054988*
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
S__inference_batch_normalization_76_layer_call_and_return_conditional_losses_705475020
.batch_normalization_76/StatefulPartitionedCall?
Relu_3Relu7batch_normalization_76/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
Relu_3?
 dense_64/StatefulPartitionedCallStatefulPartitionedCallRelu_3:activations:0dense_64_7055003dense_64_7055005*
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
E__inference_dense_64_layer_call_and_return_conditional_losses_70550022"
 dense_64/StatefulPartitionedCall?
IdentityIdentity)dense_64/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp/^batch_normalization_72/StatefulPartitionedCall/^batch_normalization_73/StatefulPartitionedCall/^batch_normalization_74/StatefulPartitionedCall/^batch_normalization_75/StatefulPartitionedCall/^batch_normalization_76/StatefulPartitionedCall!^dense_60/StatefulPartitionedCall!^dense_61/StatefulPartitionedCall!^dense_62/StatefulPartitionedCall!^dense_63/StatefulPartitionedCall!^dense_64/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????
: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_72/StatefulPartitionedCall.batch_normalization_72/StatefulPartitionedCall2`
.batch_normalization_73/StatefulPartitionedCall.batch_normalization_73/StatefulPartitionedCall2`
.batch_normalization_74/StatefulPartitionedCall.batch_normalization_74/StatefulPartitionedCall2`
.batch_normalization_75/StatefulPartitionedCall.batch_normalization_75/StatefulPartitionedCall2`
.batch_normalization_76/StatefulPartitionedCall.batch_normalization_76/StatefulPartitionedCall2D
 dense_60/StatefulPartitionedCall dense_60/StatefulPartitionedCall2D
 dense_61/StatefulPartitionedCall dense_61/StatefulPartitionedCall2D
 dense_62/StatefulPartitionedCall dense_62/StatefulPartitionedCall2D
 dense_63/StatefulPartitionedCall dense_63/StatefulPartitionedCall2D
 dense_64/StatefulPartitionedCall dense_64/StatefulPartitionedCall:J F
'
_output_shapes
:?????????


_user_specified_namex
?
?
E__inference_dense_63_layer_call_and_return_conditional_losses_7054978

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
8__inference_batch_normalization_72_layer_call_fn_7056384

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
S__inference_batch_normalization_72_layer_call_and_return_conditional_losses_70541482
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
??
?
T__inference_feed_forward_sub_net_12_layer_call_and_return_conditional_losses_7055880
xF
8batch_normalization_72_batchnorm_readvariableop_resource:
J
<batch_normalization_72_batchnorm_mul_readvariableop_resource:
H
:batch_normalization_72_batchnorm_readvariableop_1_resource:
H
:batch_normalization_72_batchnorm_readvariableop_2_resource:
9
'dense_60_matmul_readvariableop_resource:
F
8batch_normalization_73_batchnorm_readvariableop_resource:J
<batch_normalization_73_batchnorm_mul_readvariableop_resource:H
:batch_normalization_73_batchnorm_readvariableop_1_resource:H
:batch_normalization_73_batchnorm_readvariableop_2_resource:9
'dense_61_matmul_readvariableop_resource:F
8batch_normalization_74_batchnorm_readvariableop_resource:J
<batch_normalization_74_batchnorm_mul_readvariableop_resource:H
:batch_normalization_74_batchnorm_readvariableop_1_resource:H
:batch_normalization_74_batchnorm_readvariableop_2_resource:9
'dense_62_matmul_readvariableop_resource:F
8batch_normalization_75_batchnorm_readvariableop_resource:J
<batch_normalization_75_batchnorm_mul_readvariableop_resource:H
:batch_normalization_75_batchnorm_readvariableop_1_resource:H
:batch_normalization_75_batchnorm_readvariableop_2_resource:9
'dense_63_matmul_readvariableop_resource:F
8batch_normalization_76_batchnorm_readvariableop_resource:J
<batch_normalization_76_batchnorm_mul_readvariableop_resource:H
:batch_normalization_76_batchnorm_readvariableop_1_resource:H
:batch_normalization_76_batchnorm_readvariableop_2_resource:9
'dense_64_matmul_readvariableop_resource:
6
(dense_64_biasadd_readvariableop_resource:

identity??/batch_normalization_72/batchnorm/ReadVariableOp?1batch_normalization_72/batchnorm/ReadVariableOp_1?1batch_normalization_72/batchnorm/ReadVariableOp_2?3batch_normalization_72/batchnorm/mul/ReadVariableOp?/batch_normalization_73/batchnorm/ReadVariableOp?1batch_normalization_73/batchnorm/ReadVariableOp_1?1batch_normalization_73/batchnorm/ReadVariableOp_2?3batch_normalization_73/batchnorm/mul/ReadVariableOp?/batch_normalization_74/batchnorm/ReadVariableOp?1batch_normalization_74/batchnorm/ReadVariableOp_1?1batch_normalization_74/batchnorm/ReadVariableOp_2?3batch_normalization_74/batchnorm/mul/ReadVariableOp?/batch_normalization_75/batchnorm/ReadVariableOp?1batch_normalization_75/batchnorm/ReadVariableOp_1?1batch_normalization_75/batchnorm/ReadVariableOp_2?3batch_normalization_75/batchnorm/mul/ReadVariableOp?/batch_normalization_76/batchnorm/ReadVariableOp?1batch_normalization_76/batchnorm/ReadVariableOp_1?1batch_normalization_76/batchnorm/ReadVariableOp_2?3batch_normalization_76/batchnorm/mul/ReadVariableOp?dense_60/MatMul/ReadVariableOp?dense_61/MatMul/ReadVariableOp?dense_62/MatMul/ReadVariableOp?dense_63/MatMul/ReadVariableOp?dense_64/BiasAdd/ReadVariableOp?dense_64/MatMul/ReadVariableOp?
/batch_normalization_72/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_72_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype021
/batch_normalization_72/batchnorm/ReadVariableOp?
&batch_normalization_72/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2(
&batch_normalization_72/batchnorm/add/y?
$batch_normalization_72/batchnorm/addAddV27batch_normalization_72/batchnorm/ReadVariableOp:value:0/batch_normalization_72/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2&
$batch_normalization_72/batchnorm/add?
&batch_normalization_72/batchnorm/RsqrtRsqrt(batch_normalization_72/batchnorm/add:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_72/batchnorm/Rsqrt?
3batch_normalization_72/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_72_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype025
3batch_normalization_72/batchnorm/mul/ReadVariableOp?
$batch_normalization_72/batchnorm/mulMul*batch_normalization_72/batchnorm/Rsqrt:y:0;batch_normalization_72/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2&
$batch_normalization_72/batchnorm/mul?
&batch_normalization_72/batchnorm/mul_1Mulx(batch_normalization_72/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????
2(
&batch_normalization_72/batchnorm/mul_1?
1batch_normalization_72/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_72_batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype023
1batch_normalization_72/batchnorm/ReadVariableOp_1?
&batch_normalization_72/batchnorm/mul_2Mul9batch_normalization_72/batchnorm/ReadVariableOp_1:value:0(batch_normalization_72/batchnorm/mul:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_72/batchnorm/mul_2?
1batch_normalization_72/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_72_batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype023
1batch_normalization_72/batchnorm/ReadVariableOp_2?
$batch_normalization_72/batchnorm/subSub9batch_normalization_72/batchnorm/ReadVariableOp_2:value:0*batch_normalization_72/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2&
$batch_normalization_72/batchnorm/sub?
&batch_normalization_72/batchnorm/add_1AddV2*batch_normalization_72/batchnorm/mul_1:z:0(batch_normalization_72/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????
2(
&batch_normalization_72/batchnorm/add_1?
dense_60/MatMul/ReadVariableOpReadVariableOp'dense_60_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_60/MatMul/ReadVariableOp?
dense_60/MatMulMatMul*batch_normalization_72/batchnorm/add_1:z:0&dense_60/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_60/MatMul?
/batch_normalization_73/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_73_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_73/batchnorm/ReadVariableOp?
&batch_normalization_73/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2(
&batch_normalization_73/batchnorm/add/y?
$batch_normalization_73/batchnorm/addAddV27batch_normalization_73/batchnorm/ReadVariableOp:value:0/batch_normalization_73/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_73/batchnorm/add?
&batch_normalization_73/batchnorm/RsqrtRsqrt(batch_normalization_73/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_73/batchnorm/Rsqrt?
3batch_normalization_73/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_73_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_73/batchnorm/mul/ReadVariableOp?
$batch_normalization_73/batchnorm/mulMul*batch_normalization_73/batchnorm/Rsqrt:y:0;batch_normalization_73/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_73/batchnorm/mul?
&batch_normalization_73/batchnorm/mul_1Muldense_60/MatMul:product:0(batch_normalization_73/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_73/batchnorm/mul_1?
1batch_normalization_73/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_73_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_73/batchnorm/ReadVariableOp_1?
&batch_normalization_73/batchnorm/mul_2Mul9batch_normalization_73/batchnorm/ReadVariableOp_1:value:0(batch_normalization_73/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_73/batchnorm/mul_2?
1batch_normalization_73/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_73_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_73/batchnorm/ReadVariableOp_2?
$batch_normalization_73/batchnorm/subSub9batch_normalization_73/batchnorm/ReadVariableOp_2:value:0*batch_normalization_73/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_73/batchnorm/sub?
&batch_normalization_73/batchnorm/add_1AddV2*batch_normalization_73/batchnorm/mul_1:z:0(batch_normalization_73/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_73/batchnorm/add_1r
ReluRelu*batch_normalization_73/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu?
dense_61/MatMul/ReadVariableOpReadVariableOp'dense_61_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_61/MatMul/ReadVariableOp?
dense_61/MatMulMatMulRelu:activations:0&dense_61/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_61/MatMul?
/batch_normalization_74/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_74_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_74/batchnorm/ReadVariableOp?
&batch_normalization_74/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2(
&batch_normalization_74/batchnorm/add/y?
$batch_normalization_74/batchnorm/addAddV27batch_normalization_74/batchnorm/ReadVariableOp:value:0/batch_normalization_74/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_74/batchnorm/add?
&batch_normalization_74/batchnorm/RsqrtRsqrt(batch_normalization_74/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_74/batchnorm/Rsqrt?
3batch_normalization_74/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_74_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_74/batchnorm/mul/ReadVariableOp?
$batch_normalization_74/batchnorm/mulMul*batch_normalization_74/batchnorm/Rsqrt:y:0;batch_normalization_74/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_74/batchnorm/mul?
&batch_normalization_74/batchnorm/mul_1Muldense_61/MatMul:product:0(batch_normalization_74/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_74/batchnorm/mul_1?
1batch_normalization_74/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_74_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_74/batchnorm/ReadVariableOp_1?
&batch_normalization_74/batchnorm/mul_2Mul9batch_normalization_74/batchnorm/ReadVariableOp_1:value:0(batch_normalization_74/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_74/batchnorm/mul_2?
1batch_normalization_74/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_74_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_74/batchnorm/ReadVariableOp_2?
$batch_normalization_74/batchnorm/subSub9batch_normalization_74/batchnorm/ReadVariableOp_2:value:0*batch_normalization_74/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_74/batchnorm/sub?
&batch_normalization_74/batchnorm/add_1AddV2*batch_normalization_74/batchnorm/mul_1:z:0(batch_normalization_74/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_74/batchnorm/add_1v
Relu_1Relu*batch_normalization_74/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu_1?
dense_62/MatMul/ReadVariableOpReadVariableOp'dense_62_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_62/MatMul/ReadVariableOp?
dense_62/MatMulMatMulRelu_1:activations:0&dense_62/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_62/MatMul?
/batch_normalization_75/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_75_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_75/batchnorm/ReadVariableOp?
&batch_normalization_75/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2(
&batch_normalization_75/batchnorm/add/y?
$batch_normalization_75/batchnorm/addAddV27batch_normalization_75/batchnorm/ReadVariableOp:value:0/batch_normalization_75/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_75/batchnorm/add?
&batch_normalization_75/batchnorm/RsqrtRsqrt(batch_normalization_75/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_75/batchnorm/Rsqrt?
3batch_normalization_75/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_75_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_75/batchnorm/mul/ReadVariableOp?
$batch_normalization_75/batchnorm/mulMul*batch_normalization_75/batchnorm/Rsqrt:y:0;batch_normalization_75/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_75/batchnorm/mul?
&batch_normalization_75/batchnorm/mul_1Muldense_62/MatMul:product:0(batch_normalization_75/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_75/batchnorm/mul_1?
1batch_normalization_75/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_75_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_75/batchnorm/ReadVariableOp_1?
&batch_normalization_75/batchnorm/mul_2Mul9batch_normalization_75/batchnorm/ReadVariableOp_1:value:0(batch_normalization_75/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_75/batchnorm/mul_2?
1batch_normalization_75/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_75_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_75/batchnorm/ReadVariableOp_2?
$batch_normalization_75/batchnorm/subSub9batch_normalization_75/batchnorm/ReadVariableOp_2:value:0*batch_normalization_75/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_75/batchnorm/sub?
&batch_normalization_75/batchnorm/add_1AddV2*batch_normalization_75/batchnorm/mul_1:z:0(batch_normalization_75/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_75/batchnorm/add_1v
Relu_2Relu*batch_normalization_75/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu_2?
dense_63/MatMul/ReadVariableOpReadVariableOp'dense_63_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_63/MatMul/ReadVariableOp?
dense_63/MatMulMatMulRelu_2:activations:0&dense_63/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_63/MatMul?
/batch_normalization_76/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_76_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_76/batchnorm/ReadVariableOp?
&batch_normalization_76/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2(
&batch_normalization_76/batchnorm/add/y?
$batch_normalization_76/batchnorm/addAddV27batch_normalization_76/batchnorm/ReadVariableOp:value:0/batch_normalization_76/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_76/batchnorm/add?
&batch_normalization_76/batchnorm/RsqrtRsqrt(batch_normalization_76/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_76/batchnorm/Rsqrt?
3batch_normalization_76/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_76_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_76/batchnorm/mul/ReadVariableOp?
$batch_normalization_76/batchnorm/mulMul*batch_normalization_76/batchnorm/Rsqrt:y:0;batch_normalization_76/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_76/batchnorm/mul?
&batch_normalization_76/batchnorm/mul_1Muldense_63/MatMul:product:0(batch_normalization_76/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_76/batchnorm/mul_1?
1batch_normalization_76/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_76_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_76/batchnorm/ReadVariableOp_1?
&batch_normalization_76/batchnorm/mul_2Mul9batch_normalization_76/batchnorm/ReadVariableOp_1:value:0(batch_normalization_76/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_76/batchnorm/mul_2?
1batch_normalization_76/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_76_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_76/batchnorm/ReadVariableOp_2?
$batch_normalization_76/batchnorm/subSub9batch_normalization_76/batchnorm/ReadVariableOp_2:value:0*batch_normalization_76/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_76/batchnorm/sub?
&batch_normalization_76/batchnorm/add_1AddV2*batch_normalization_76/batchnorm/mul_1:z:0(batch_normalization_76/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_76/batchnorm/add_1v
Relu_3Relu*batch_normalization_76/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu_3?
dense_64/MatMul/ReadVariableOpReadVariableOp'dense_64_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_64/MatMul/ReadVariableOp?
dense_64/MatMulMatMulRelu_3:activations:0&dense_64/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_64/MatMul?
dense_64/BiasAdd/ReadVariableOpReadVariableOp(dense_64_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_64/BiasAdd/ReadVariableOp?
dense_64/BiasAddBiasAdddense_64/MatMul:product:0'dense_64/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_64/BiasAddt
IdentityIdentitydense_64/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?

NoOpNoOp0^batch_normalization_72/batchnorm/ReadVariableOp2^batch_normalization_72/batchnorm/ReadVariableOp_12^batch_normalization_72/batchnorm/ReadVariableOp_24^batch_normalization_72/batchnorm/mul/ReadVariableOp0^batch_normalization_73/batchnorm/ReadVariableOp2^batch_normalization_73/batchnorm/ReadVariableOp_12^batch_normalization_73/batchnorm/ReadVariableOp_24^batch_normalization_73/batchnorm/mul/ReadVariableOp0^batch_normalization_74/batchnorm/ReadVariableOp2^batch_normalization_74/batchnorm/ReadVariableOp_12^batch_normalization_74/batchnorm/ReadVariableOp_24^batch_normalization_74/batchnorm/mul/ReadVariableOp0^batch_normalization_75/batchnorm/ReadVariableOp2^batch_normalization_75/batchnorm/ReadVariableOp_12^batch_normalization_75/batchnorm/ReadVariableOp_24^batch_normalization_75/batchnorm/mul/ReadVariableOp0^batch_normalization_76/batchnorm/ReadVariableOp2^batch_normalization_76/batchnorm/ReadVariableOp_12^batch_normalization_76/batchnorm/ReadVariableOp_24^batch_normalization_76/batchnorm/mul/ReadVariableOp^dense_60/MatMul/ReadVariableOp^dense_61/MatMul/ReadVariableOp^dense_62/MatMul/ReadVariableOp^dense_63/MatMul/ReadVariableOp ^dense_64/BiasAdd/ReadVariableOp^dense_64/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????
: : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_72/batchnorm/ReadVariableOp/batch_normalization_72/batchnorm/ReadVariableOp2f
1batch_normalization_72/batchnorm/ReadVariableOp_11batch_normalization_72/batchnorm/ReadVariableOp_12f
1batch_normalization_72/batchnorm/ReadVariableOp_21batch_normalization_72/batchnorm/ReadVariableOp_22j
3batch_normalization_72/batchnorm/mul/ReadVariableOp3batch_normalization_72/batchnorm/mul/ReadVariableOp2b
/batch_normalization_73/batchnorm/ReadVariableOp/batch_normalization_73/batchnorm/ReadVariableOp2f
1batch_normalization_73/batchnorm/ReadVariableOp_11batch_normalization_73/batchnorm/ReadVariableOp_12f
1batch_normalization_73/batchnorm/ReadVariableOp_21batch_normalization_73/batchnorm/ReadVariableOp_22j
3batch_normalization_73/batchnorm/mul/ReadVariableOp3batch_normalization_73/batchnorm/mul/ReadVariableOp2b
/batch_normalization_74/batchnorm/ReadVariableOp/batch_normalization_74/batchnorm/ReadVariableOp2f
1batch_normalization_74/batchnorm/ReadVariableOp_11batch_normalization_74/batchnorm/ReadVariableOp_12f
1batch_normalization_74/batchnorm/ReadVariableOp_21batch_normalization_74/batchnorm/ReadVariableOp_22j
3batch_normalization_74/batchnorm/mul/ReadVariableOp3batch_normalization_74/batchnorm/mul/ReadVariableOp2b
/batch_normalization_75/batchnorm/ReadVariableOp/batch_normalization_75/batchnorm/ReadVariableOp2f
1batch_normalization_75/batchnorm/ReadVariableOp_11batch_normalization_75/batchnorm/ReadVariableOp_12f
1batch_normalization_75/batchnorm/ReadVariableOp_21batch_normalization_75/batchnorm/ReadVariableOp_22j
3batch_normalization_75/batchnorm/mul/ReadVariableOp3batch_normalization_75/batchnorm/mul/ReadVariableOp2b
/batch_normalization_76/batchnorm/ReadVariableOp/batch_normalization_76/batchnorm/ReadVariableOp2f
1batch_normalization_76/batchnorm/ReadVariableOp_11batch_normalization_76/batchnorm/ReadVariableOp_12f
1batch_normalization_76/batchnorm/ReadVariableOp_21batch_normalization_76/batchnorm/ReadVariableOp_22j
3batch_normalization_76/batchnorm/mul/ReadVariableOp3batch_normalization_76/batchnorm/mul/ReadVariableOp2@
dense_60/MatMul/ReadVariableOpdense_60/MatMul/ReadVariableOp2@
dense_61/MatMul/ReadVariableOpdense_61/MatMul/ReadVariableOp2@
dense_62/MatMul/ReadVariableOpdense_62/MatMul/ReadVariableOp2@
dense_63/MatMul/ReadVariableOpdense_63/MatMul/ReadVariableOp2B
dense_64/BiasAdd/ReadVariableOpdense_64/BiasAdd/ReadVariableOp2@
dense_64/MatMul/ReadVariableOpdense_64/MatMul/ReadVariableOp:J F
'
_output_shapes
:?????????


_user_specified_namex
?
?
E__inference_dense_60_layer_call_and_return_conditional_losses_7054915

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
S__inference_batch_normalization_72_layer_call_and_return_conditional_losses_7056404

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
?
?
E__inference_dense_61_layer_call_and_return_conditional_losses_7056796

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
S__inference_batch_normalization_73_layer_call_and_return_conditional_losses_7056522

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
E__inference_dense_63_layer_call_and_return_conditional_losses_7056824

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
8__inference_batch_normalization_72_layer_call_fn_7056371

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
S__inference_batch_normalization_72_layer_call_and_return_conditional_losses_70540862
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
S__inference_batch_normalization_76_layer_call_and_return_conditional_losses_7056768

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
?,
?
S__inference_batch_normalization_72_layer_call_and_return_conditional_losses_7056440

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
??
?
T__inference_feed_forward_sub_net_12_layer_call_and_return_conditional_losses_7056172
input_1F
8batch_normalization_72_batchnorm_readvariableop_resource:
J
<batch_normalization_72_batchnorm_mul_readvariableop_resource:
H
:batch_normalization_72_batchnorm_readvariableop_1_resource:
H
:batch_normalization_72_batchnorm_readvariableop_2_resource:
9
'dense_60_matmul_readvariableop_resource:
F
8batch_normalization_73_batchnorm_readvariableop_resource:J
<batch_normalization_73_batchnorm_mul_readvariableop_resource:H
:batch_normalization_73_batchnorm_readvariableop_1_resource:H
:batch_normalization_73_batchnorm_readvariableop_2_resource:9
'dense_61_matmul_readvariableop_resource:F
8batch_normalization_74_batchnorm_readvariableop_resource:J
<batch_normalization_74_batchnorm_mul_readvariableop_resource:H
:batch_normalization_74_batchnorm_readvariableop_1_resource:H
:batch_normalization_74_batchnorm_readvariableop_2_resource:9
'dense_62_matmul_readvariableop_resource:F
8batch_normalization_75_batchnorm_readvariableop_resource:J
<batch_normalization_75_batchnorm_mul_readvariableop_resource:H
:batch_normalization_75_batchnorm_readvariableop_1_resource:H
:batch_normalization_75_batchnorm_readvariableop_2_resource:9
'dense_63_matmul_readvariableop_resource:F
8batch_normalization_76_batchnorm_readvariableop_resource:J
<batch_normalization_76_batchnorm_mul_readvariableop_resource:H
:batch_normalization_76_batchnorm_readvariableop_1_resource:H
:batch_normalization_76_batchnorm_readvariableop_2_resource:9
'dense_64_matmul_readvariableop_resource:
6
(dense_64_biasadd_readvariableop_resource:

identity??/batch_normalization_72/batchnorm/ReadVariableOp?1batch_normalization_72/batchnorm/ReadVariableOp_1?1batch_normalization_72/batchnorm/ReadVariableOp_2?3batch_normalization_72/batchnorm/mul/ReadVariableOp?/batch_normalization_73/batchnorm/ReadVariableOp?1batch_normalization_73/batchnorm/ReadVariableOp_1?1batch_normalization_73/batchnorm/ReadVariableOp_2?3batch_normalization_73/batchnorm/mul/ReadVariableOp?/batch_normalization_74/batchnorm/ReadVariableOp?1batch_normalization_74/batchnorm/ReadVariableOp_1?1batch_normalization_74/batchnorm/ReadVariableOp_2?3batch_normalization_74/batchnorm/mul/ReadVariableOp?/batch_normalization_75/batchnorm/ReadVariableOp?1batch_normalization_75/batchnorm/ReadVariableOp_1?1batch_normalization_75/batchnorm/ReadVariableOp_2?3batch_normalization_75/batchnorm/mul/ReadVariableOp?/batch_normalization_76/batchnorm/ReadVariableOp?1batch_normalization_76/batchnorm/ReadVariableOp_1?1batch_normalization_76/batchnorm/ReadVariableOp_2?3batch_normalization_76/batchnorm/mul/ReadVariableOp?dense_60/MatMul/ReadVariableOp?dense_61/MatMul/ReadVariableOp?dense_62/MatMul/ReadVariableOp?dense_63/MatMul/ReadVariableOp?dense_64/BiasAdd/ReadVariableOp?dense_64/MatMul/ReadVariableOp?
/batch_normalization_72/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_72_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype021
/batch_normalization_72/batchnorm/ReadVariableOp?
&batch_normalization_72/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2(
&batch_normalization_72/batchnorm/add/y?
$batch_normalization_72/batchnorm/addAddV27batch_normalization_72/batchnorm/ReadVariableOp:value:0/batch_normalization_72/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2&
$batch_normalization_72/batchnorm/add?
&batch_normalization_72/batchnorm/RsqrtRsqrt(batch_normalization_72/batchnorm/add:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_72/batchnorm/Rsqrt?
3batch_normalization_72/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_72_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype025
3batch_normalization_72/batchnorm/mul/ReadVariableOp?
$batch_normalization_72/batchnorm/mulMul*batch_normalization_72/batchnorm/Rsqrt:y:0;batch_normalization_72/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2&
$batch_normalization_72/batchnorm/mul?
&batch_normalization_72/batchnorm/mul_1Mulinput_1(batch_normalization_72/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????
2(
&batch_normalization_72/batchnorm/mul_1?
1batch_normalization_72/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_72_batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype023
1batch_normalization_72/batchnorm/ReadVariableOp_1?
&batch_normalization_72/batchnorm/mul_2Mul9batch_normalization_72/batchnorm/ReadVariableOp_1:value:0(batch_normalization_72/batchnorm/mul:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_72/batchnorm/mul_2?
1batch_normalization_72/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_72_batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype023
1batch_normalization_72/batchnorm/ReadVariableOp_2?
$batch_normalization_72/batchnorm/subSub9batch_normalization_72/batchnorm/ReadVariableOp_2:value:0*batch_normalization_72/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2&
$batch_normalization_72/batchnorm/sub?
&batch_normalization_72/batchnorm/add_1AddV2*batch_normalization_72/batchnorm/mul_1:z:0(batch_normalization_72/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????
2(
&batch_normalization_72/batchnorm/add_1?
dense_60/MatMul/ReadVariableOpReadVariableOp'dense_60_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_60/MatMul/ReadVariableOp?
dense_60/MatMulMatMul*batch_normalization_72/batchnorm/add_1:z:0&dense_60/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_60/MatMul?
/batch_normalization_73/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_73_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_73/batchnorm/ReadVariableOp?
&batch_normalization_73/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2(
&batch_normalization_73/batchnorm/add/y?
$batch_normalization_73/batchnorm/addAddV27batch_normalization_73/batchnorm/ReadVariableOp:value:0/batch_normalization_73/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_73/batchnorm/add?
&batch_normalization_73/batchnorm/RsqrtRsqrt(batch_normalization_73/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_73/batchnorm/Rsqrt?
3batch_normalization_73/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_73_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_73/batchnorm/mul/ReadVariableOp?
$batch_normalization_73/batchnorm/mulMul*batch_normalization_73/batchnorm/Rsqrt:y:0;batch_normalization_73/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_73/batchnorm/mul?
&batch_normalization_73/batchnorm/mul_1Muldense_60/MatMul:product:0(batch_normalization_73/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_73/batchnorm/mul_1?
1batch_normalization_73/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_73_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_73/batchnorm/ReadVariableOp_1?
&batch_normalization_73/batchnorm/mul_2Mul9batch_normalization_73/batchnorm/ReadVariableOp_1:value:0(batch_normalization_73/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_73/batchnorm/mul_2?
1batch_normalization_73/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_73_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_73/batchnorm/ReadVariableOp_2?
$batch_normalization_73/batchnorm/subSub9batch_normalization_73/batchnorm/ReadVariableOp_2:value:0*batch_normalization_73/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_73/batchnorm/sub?
&batch_normalization_73/batchnorm/add_1AddV2*batch_normalization_73/batchnorm/mul_1:z:0(batch_normalization_73/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_73/batchnorm/add_1r
ReluRelu*batch_normalization_73/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu?
dense_61/MatMul/ReadVariableOpReadVariableOp'dense_61_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_61/MatMul/ReadVariableOp?
dense_61/MatMulMatMulRelu:activations:0&dense_61/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_61/MatMul?
/batch_normalization_74/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_74_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_74/batchnorm/ReadVariableOp?
&batch_normalization_74/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2(
&batch_normalization_74/batchnorm/add/y?
$batch_normalization_74/batchnorm/addAddV27batch_normalization_74/batchnorm/ReadVariableOp:value:0/batch_normalization_74/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_74/batchnorm/add?
&batch_normalization_74/batchnorm/RsqrtRsqrt(batch_normalization_74/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_74/batchnorm/Rsqrt?
3batch_normalization_74/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_74_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_74/batchnorm/mul/ReadVariableOp?
$batch_normalization_74/batchnorm/mulMul*batch_normalization_74/batchnorm/Rsqrt:y:0;batch_normalization_74/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_74/batchnorm/mul?
&batch_normalization_74/batchnorm/mul_1Muldense_61/MatMul:product:0(batch_normalization_74/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_74/batchnorm/mul_1?
1batch_normalization_74/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_74_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_74/batchnorm/ReadVariableOp_1?
&batch_normalization_74/batchnorm/mul_2Mul9batch_normalization_74/batchnorm/ReadVariableOp_1:value:0(batch_normalization_74/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_74/batchnorm/mul_2?
1batch_normalization_74/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_74_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_74/batchnorm/ReadVariableOp_2?
$batch_normalization_74/batchnorm/subSub9batch_normalization_74/batchnorm/ReadVariableOp_2:value:0*batch_normalization_74/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_74/batchnorm/sub?
&batch_normalization_74/batchnorm/add_1AddV2*batch_normalization_74/batchnorm/mul_1:z:0(batch_normalization_74/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_74/batchnorm/add_1v
Relu_1Relu*batch_normalization_74/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu_1?
dense_62/MatMul/ReadVariableOpReadVariableOp'dense_62_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_62/MatMul/ReadVariableOp?
dense_62/MatMulMatMulRelu_1:activations:0&dense_62/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_62/MatMul?
/batch_normalization_75/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_75_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_75/batchnorm/ReadVariableOp?
&batch_normalization_75/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2(
&batch_normalization_75/batchnorm/add/y?
$batch_normalization_75/batchnorm/addAddV27batch_normalization_75/batchnorm/ReadVariableOp:value:0/batch_normalization_75/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_75/batchnorm/add?
&batch_normalization_75/batchnorm/RsqrtRsqrt(batch_normalization_75/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_75/batchnorm/Rsqrt?
3batch_normalization_75/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_75_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_75/batchnorm/mul/ReadVariableOp?
$batch_normalization_75/batchnorm/mulMul*batch_normalization_75/batchnorm/Rsqrt:y:0;batch_normalization_75/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_75/batchnorm/mul?
&batch_normalization_75/batchnorm/mul_1Muldense_62/MatMul:product:0(batch_normalization_75/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_75/batchnorm/mul_1?
1batch_normalization_75/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_75_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_75/batchnorm/ReadVariableOp_1?
&batch_normalization_75/batchnorm/mul_2Mul9batch_normalization_75/batchnorm/ReadVariableOp_1:value:0(batch_normalization_75/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_75/batchnorm/mul_2?
1batch_normalization_75/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_75_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_75/batchnorm/ReadVariableOp_2?
$batch_normalization_75/batchnorm/subSub9batch_normalization_75/batchnorm/ReadVariableOp_2:value:0*batch_normalization_75/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_75/batchnorm/sub?
&batch_normalization_75/batchnorm/add_1AddV2*batch_normalization_75/batchnorm/mul_1:z:0(batch_normalization_75/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_75/batchnorm/add_1v
Relu_2Relu*batch_normalization_75/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu_2?
dense_63/MatMul/ReadVariableOpReadVariableOp'dense_63_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_63/MatMul/ReadVariableOp?
dense_63/MatMulMatMulRelu_2:activations:0&dense_63/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_63/MatMul?
/batch_normalization_76/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_76_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_76/batchnorm/ReadVariableOp?
&batch_normalization_76/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2(
&batch_normalization_76/batchnorm/add/y?
$batch_normalization_76/batchnorm/addAddV27batch_normalization_76/batchnorm/ReadVariableOp:value:0/batch_normalization_76/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_76/batchnorm/add?
&batch_normalization_76/batchnorm/RsqrtRsqrt(batch_normalization_76/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_76/batchnorm/Rsqrt?
3batch_normalization_76/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_76_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_76/batchnorm/mul/ReadVariableOp?
$batch_normalization_76/batchnorm/mulMul*batch_normalization_76/batchnorm/Rsqrt:y:0;batch_normalization_76/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_76/batchnorm/mul?
&batch_normalization_76/batchnorm/mul_1Muldense_63/MatMul:product:0(batch_normalization_76/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_76/batchnorm/mul_1?
1batch_normalization_76/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_76_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_76/batchnorm/ReadVariableOp_1?
&batch_normalization_76/batchnorm/mul_2Mul9batch_normalization_76/batchnorm/ReadVariableOp_1:value:0(batch_normalization_76/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_76/batchnorm/mul_2?
1batch_normalization_76/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_76_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_76/batchnorm/ReadVariableOp_2?
$batch_normalization_76/batchnorm/subSub9batch_normalization_76/batchnorm/ReadVariableOp_2:value:0*batch_normalization_76/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_76/batchnorm/sub?
&batch_normalization_76/batchnorm/add_1AddV2*batch_normalization_76/batchnorm/mul_1:z:0(batch_normalization_76/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_76/batchnorm/add_1v
Relu_3Relu*batch_normalization_76/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu_3?
dense_64/MatMul/ReadVariableOpReadVariableOp'dense_64_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_64/MatMul/ReadVariableOp?
dense_64/MatMulMatMulRelu_3:activations:0&dense_64/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_64/MatMul?
dense_64/BiasAdd/ReadVariableOpReadVariableOp(dense_64_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_64/BiasAdd/ReadVariableOp?
dense_64/BiasAddBiasAdddense_64/MatMul:product:0'dense_64/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_64/BiasAddt
IdentityIdentitydense_64/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?

NoOpNoOp0^batch_normalization_72/batchnorm/ReadVariableOp2^batch_normalization_72/batchnorm/ReadVariableOp_12^batch_normalization_72/batchnorm/ReadVariableOp_24^batch_normalization_72/batchnorm/mul/ReadVariableOp0^batch_normalization_73/batchnorm/ReadVariableOp2^batch_normalization_73/batchnorm/ReadVariableOp_12^batch_normalization_73/batchnorm/ReadVariableOp_24^batch_normalization_73/batchnorm/mul/ReadVariableOp0^batch_normalization_74/batchnorm/ReadVariableOp2^batch_normalization_74/batchnorm/ReadVariableOp_12^batch_normalization_74/batchnorm/ReadVariableOp_24^batch_normalization_74/batchnorm/mul/ReadVariableOp0^batch_normalization_75/batchnorm/ReadVariableOp2^batch_normalization_75/batchnorm/ReadVariableOp_12^batch_normalization_75/batchnorm/ReadVariableOp_24^batch_normalization_75/batchnorm/mul/ReadVariableOp0^batch_normalization_76/batchnorm/ReadVariableOp2^batch_normalization_76/batchnorm/ReadVariableOp_12^batch_normalization_76/batchnorm/ReadVariableOp_24^batch_normalization_76/batchnorm/mul/ReadVariableOp^dense_60/MatMul/ReadVariableOp^dense_61/MatMul/ReadVariableOp^dense_62/MatMul/ReadVariableOp^dense_63/MatMul/ReadVariableOp ^dense_64/BiasAdd/ReadVariableOp^dense_64/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????
: : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_72/batchnorm/ReadVariableOp/batch_normalization_72/batchnorm/ReadVariableOp2f
1batch_normalization_72/batchnorm/ReadVariableOp_11batch_normalization_72/batchnorm/ReadVariableOp_12f
1batch_normalization_72/batchnorm/ReadVariableOp_21batch_normalization_72/batchnorm/ReadVariableOp_22j
3batch_normalization_72/batchnorm/mul/ReadVariableOp3batch_normalization_72/batchnorm/mul/ReadVariableOp2b
/batch_normalization_73/batchnorm/ReadVariableOp/batch_normalization_73/batchnorm/ReadVariableOp2f
1batch_normalization_73/batchnorm/ReadVariableOp_11batch_normalization_73/batchnorm/ReadVariableOp_12f
1batch_normalization_73/batchnorm/ReadVariableOp_21batch_normalization_73/batchnorm/ReadVariableOp_22j
3batch_normalization_73/batchnorm/mul/ReadVariableOp3batch_normalization_73/batchnorm/mul/ReadVariableOp2b
/batch_normalization_74/batchnorm/ReadVariableOp/batch_normalization_74/batchnorm/ReadVariableOp2f
1batch_normalization_74/batchnorm/ReadVariableOp_11batch_normalization_74/batchnorm/ReadVariableOp_12f
1batch_normalization_74/batchnorm/ReadVariableOp_21batch_normalization_74/batchnorm/ReadVariableOp_22j
3batch_normalization_74/batchnorm/mul/ReadVariableOp3batch_normalization_74/batchnorm/mul/ReadVariableOp2b
/batch_normalization_75/batchnorm/ReadVariableOp/batch_normalization_75/batchnorm/ReadVariableOp2f
1batch_normalization_75/batchnorm/ReadVariableOp_11batch_normalization_75/batchnorm/ReadVariableOp_12f
1batch_normalization_75/batchnorm/ReadVariableOp_21batch_normalization_75/batchnorm/ReadVariableOp_22j
3batch_normalization_75/batchnorm/mul/ReadVariableOp3batch_normalization_75/batchnorm/mul/ReadVariableOp2b
/batch_normalization_76/batchnorm/ReadVariableOp/batch_normalization_76/batchnorm/ReadVariableOp2f
1batch_normalization_76/batchnorm/ReadVariableOp_11batch_normalization_76/batchnorm/ReadVariableOp_12f
1batch_normalization_76/batchnorm/ReadVariableOp_21batch_normalization_76/batchnorm/ReadVariableOp_22j
3batch_normalization_76/batchnorm/mul/ReadVariableOp3batch_normalization_76/batchnorm/mul/ReadVariableOp2@
dense_60/MatMul/ReadVariableOpdense_60/MatMul/ReadVariableOp2@
dense_61/MatMul/ReadVariableOpdense_61/MatMul/ReadVariableOp2@
dense_62/MatMul/ReadVariableOpdense_62/MatMul/ReadVariableOp2@
dense_63/MatMul/ReadVariableOpdense_63/MatMul/ReadVariableOp2B
dense_64/BiasAdd/ReadVariableOpdense_64/BiasAdd/ReadVariableOp2@
dense_64/MatMul/ReadVariableOpdense_64/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????

!
_user_specified_name	input_1
??
?
#__inference__traced_restore_7057032
file_prefixe
Wassignvariableop_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_72_gamma:
f
Xassignvariableop_1_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_72_beta:
g
Yassignvariableop_2_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_73_gamma:f
Xassignvariableop_3_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_73_beta:g
Yassignvariableop_4_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_74_gamma:f
Xassignvariableop_5_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_74_beta:g
Yassignvariableop_6_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_75_gamma:f
Xassignvariableop_7_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_75_beta:g
Yassignvariableop_8_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_76_gamma:f
Xassignvariableop_9_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_76_beta:n
`assignvariableop_10_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_72_moving_mean:
r
dassignvariableop_11_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_72_moving_variance:
n
`assignvariableop_12_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_73_moving_mean:r
dassignvariableop_13_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_73_moving_variance:n
`assignvariableop_14_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_74_moving_mean:r
dassignvariableop_15_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_74_moving_variance:n
`assignvariableop_16_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_75_moving_mean:r
dassignvariableop_17_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_75_moving_variance:n
`assignvariableop_18_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_76_moving_mean:r
dassignvariableop_19_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_76_moving_variance:_
Massignvariableop_20_nonshared_model_1_feed_forward_sub_net_12_dense_60_kernel:
_
Massignvariableop_21_nonshared_model_1_feed_forward_sub_net_12_dense_61_kernel:_
Massignvariableop_22_nonshared_model_1_feed_forward_sub_net_12_dense_62_kernel:_
Massignvariableop_23_nonshared_model_1_feed_forward_sub_net_12_dense_63_kernel:_
Massignvariableop_24_nonshared_model_1_feed_forward_sub_net_12_dense_64_kernel:
Y
Kassignvariableop_25_nonshared_model_1_feed_forward_sub_net_12_dense_64_bias:
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
AssignVariableOpAssignVariableOpWassignvariableop_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_72_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpXassignvariableop_1_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_72_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpYassignvariableop_2_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_73_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpXassignvariableop_3_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_73_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpYassignvariableop_4_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_74_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpXassignvariableop_5_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_74_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpYassignvariableop_6_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_75_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpXassignvariableop_7_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_75_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpYassignvariableop_8_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_76_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpXassignvariableop_9_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_76_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp`assignvariableop_10_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_72_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpdassignvariableop_11_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_72_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp`assignvariableop_12_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_73_moving_meanIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpdassignvariableop_13_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_73_moving_varianceIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp`assignvariableop_14_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_74_moving_meanIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpdassignvariableop_15_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_74_moving_varianceIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp`assignvariableop_16_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_75_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpdassignvariableop_17_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_75_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp`assignvariableop_18_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_76_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpdassignvariableop_19_nonshared_model_1_feed_forward_sub_net_12_batch_normalization_76_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpMassignvariableop_20_nonshared_model_1_feed_forward_sub_net_12_dense_60_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpMassignvariableop_21_nonshared_model_1_feed_forward_sub_net_12_dense_61_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpMassignvariableop_22_nonshared_model_1_feed_forward_sub_net_12_dense_62_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpMassignvariableop_23_nonshared_model_1_feed_forward_sub_net_12_dense_63_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpMassignvariableop_24_nonshared_model_1_feed_forward_sub_net_12_dense_64_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpKassignvariableop_25_nonshared_model_1_feed_forward_sub_net_12_dense_64_biasIdentity_25:output:0"/device:CPU:0*
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
?
?
S__inference_batch_normalization_76_layer_call_and_return_conditional_losses_7054750

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
?,
?
S__inference_batch_normalization_75_layer_call_and_return_conditional_losses_7056686

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
9__inference_feed_forward_sub_net_12_layer_call_fn_7055774
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
T__inference_feed_forward_sub_net_12_layer_call_and_return_conditional_losses_70552352
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
?
?
8__inference_batch_normalization_76_layer_call_fn_7056699

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
S__inference_batch_normalization_76_layer_call_and_return_conditional_losses_70547502
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
S__inference_batch_normalization_73_layer_call_and_return_conditional_losses_7054314

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
S__inference_batch_normalization_74_layer_call_and_return_conditional_losses_7056568

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
*__inference_dense_61_layer_call_fn_7056789

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
E__inference_dense_61_layer_call_and_return_conditional_losses_70549362
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
?E
?
T__inference_feed_forward_sub_net_12_layer_call_and_return_conditional_losses_7055235
x,
batch_normalization_72_7055168:
,
batch_normalization_72_7055170:
,
batch_normalization_72_7055172:
,
batch_normalization_72_7055174:
"
dense_60_7055177:
,
batch_normalization_73_7055180:,
batch_normalization_73_7055182:,
batch_normalization_73_7055184:,
batch_normalization_73_7055186:"
dense_61_7055190:,
batch_normalization_74_7055193:,
batch_normalization_74_7055195:,
batch_normalization_74_7055197:,
batch_normalization_74_7055199:"
dense_62_7055203:,
batch_normalization_75_7055206:,
batch_normalization_75_7055208:,
batch_normalization_75_7055210:,
batch_normalization_75_7055212:"
dense_63_7055216:,
batch_normalization_76_7055219:,
batch_normalization_76_7055221:,
batch_normalization_76_7055223:,
batch_normalization_76_7055225:"
dense_64_7055229:

dense_64_7055231:

identity??.batch_normalization_72/StatefulPartitionedCall?.batch_normalization_73/StatefulPartitionedCall?.batch_normalization_74/StatefulPartitionedCall?.batch_normalization_75/StatefulPartitionedCall?.batch_normalization_76/StatefulPartitionedCall? dense_60/StatefulPartitionedCall? dense_61/StatefulPartitionedCall? dense_62/StatefulPartitionedCall? dense_63/StatefulPartitionedCall? dense_64/StatefulPartitionedCall?
.batch_normalization_72/StatefulPartitionedCallStatefulPartitionedCallxbatch_normalization_72_7055168batch_normalization_72_7055170batch_normalization_72_7055172batch_normalization_72_7055174*
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
S__inference_batch_normalization_72_layer_call_and_return_conditional_losses_705414820
.batch_normalization_72/StatefulPartitionedCall?
 dense_60/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_72/StatefulPartitionedCall:output:0dense_60_7055177*
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
E__inference_dense_60_layer_call_and_return_conditional_losses_70549152"
 dense_60/StatefulPartitionedCall?
.batch_normalization_73/StatefulPartitionedCallStatefulPartitionedCall)dense_60/StatefulPartitionedCall:output:0batch_normalization_73_7055180batch_normalization_73_7055182batch_normalization_73_7055184batch_normalization_73_7055186*
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
S__inference_batch_normalization_73_layer_call_and_return_conditional_losses_705431420
.batch_normalization_73/StatefulPartitionedCall
ReluRelu7batch_normalization_73/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
Relu?
 dense_61/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0dense_61_7055190*
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
E__inference_dense_61_layer_call_and_return_conditional_losses_70549362"
 dense_61/StatefulPartitionedCall?
.batch_normalization_74/StatefulPartitionedCallStatefulPartitionedCall)dense_61/StatefulPartitionedCall:output:0batch_normalization_74_7055193batch_normalization_74_7055195batch_normalization_74_7055197batch_normalization_74_7055199*
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
S__inference_batch_normalization_74_layer_call_and_return_conditional_losses_705448020
.batch_normalization_74/StatefulPartitionedCall?
Relu_1Relu7batch_normalization_74/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
Relu_1?
 dense_62/StatefulPartitionedCallStatefulPartitionedCallRelu_1:activations:0dense_62_7055203*
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
E__inference_dense_62_layer_call_and_return_conditional_losses_70549572"
 dense_62/StatefulPartitionedCall?
.batch_normalization_75/StatefulPartitionedCallStatefulPartitionedCall)dense_62/StatefulPartitionedCall:output:0batch_normalization_75_7055206batch_normalization_75_7055208batch_normalization_75_7055210batch_normalization_75_7055212*
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
S__inference_batch_normalization_75_layer_call_and_return_conditional_losses_705464620
.batch_normalization_75/StatefulPartitionedCall?
Relu_2Relu7batch_normalization_75/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
Relu_2?
 dense_63/StatefulPartitionedCallStatefulPartitionedCallRelu_2:activations:0dense_63_7055216*
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
E__inference_dense_63_layer_call_and_return_conditional_losses_70549782"
 dense_63/StatefulPartitionedCall?
.batch_normalization_76/StatefulPartitionedCallStatefulPartitionedCall)dense_63/StatefulPartitionedCall:output:0batch_normalization_76_7055219batch_normalization_76_7055221batch_normalization_76_7055223batch_normalization_76_7055225*
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
S__inference_batch_normalization_76_layer_call_and_return_conditional_losses_705481220
.batch_normalization_76/StatefulPartitionedCall?
Relu_3Relu7batch_normalization_76/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
Relu_3?
 dense_64/StatefulPartitionedCallStatefulPartitionedCallRelu_3:activations:0dense_64_7055229dense_64_7055231*
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
E__inference_dense_64_layer_call_and_return_conditional_losses_70550022"
 dense_64/StatefulPartitionedCall?
IdentityIdentity)dense_64/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp/^batch_normalization_72/StatefulPartitionedCall/^batch_normalization_73/StatefulPartitionedCall/^batch_normalization_74/StatefulPartitionedCall/^batch_normalization_75/StatefulPartitionedCall/^batch_normalization_76/StatefulPartitionedCall!^dense_60/StatefulPartitionedCall!^dense_61/StatefulPartitionedCall!^dense_62/StatefulPartitionedCall!^dense_63/StatefulPartitionedCall!^dense_64/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????
: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_72/StatefulPartitionedCall.batch_normalization_72/StatefulPartitionedCall2`
.batch_normalization_73/StatefulPartitionedCall.batch_normalization_73/StatefulPartitionedCall2`
.batch_normalization_74/StatefulPartitionedCall.batch_normalization_74/StatefulPartitionedCall2`
.batch_normalization_75/StatefulPartitionedCall.batch_normalization_75/StatefulPartitionedCall2`
.batch_normalization_76/StatefulPartitionedCall.batch_normalization_76/StatefulPartitionedCall2D
 dense_60/StatefulPartitionedCall dense_60/StatefulPartitionedCall2D
 dense_61/StatefulPartitionedCall dense_61/StatefulPartitionedCall2D
 dense_62/StatefulPartitionedCall dense_62/StatefulPartitionedCall2D
 dense_63/StatefulPartitionedCall dense_63/StatefulPartitionedCall2D
 dense_64/StatefulPartitionedCall dense_64/StatefulPartitionedCall:J F
'
_output_shapes
:?????????


_user_specified_namex
?
?
E__inference_dense_60_layer_call_and_return_conditional_losses_7056782

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
?,
?
S__inference_batch_normalization_74_layer_call_and_return_conditional_losses_7054480

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
8__inference_batch_normalization_76_layer_call_fn_7056712

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
S__inference_batch_normalization_76_layer_call_and_return_conditional_losses_70548122
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
S__inference_batch_normalization_73_layer_call_and_return_conditional_losses_7054252

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
?
*__inference_dense_64_layer_call_fn_7056833

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
E__inference_dense_64_layer_call_and_return_conditional_losses_70550022
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
?
?
9__inference_feed_forward_sub_net_12_layer_call_fn_7055603
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
T__inference_feed_forward_sub_net_12_layer_call_and_return_conditional_losses_70550092
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
E__inference_dense_62_layer_call_and_return_conditional_losses_7056810

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
S__inference_batch_normalization_74_layer_call_and_return_conditional_losses_7056604

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
8__inference_batch_normalization_74_layer_call_fn_7056535

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
S__inference_batch_normalization_74_layer_call_and_return_conditional_losses_70544182
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
?
~
*__inference_dense_63_layer_call_fn_7056817

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
E__inference_dense_63_layer_call_and_return_conditional_losses_70549782
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
S__inference_batch_normalization_72_layer_call_and_return_conditional_losses_7054148

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
?
~
*__inference_dense_60_layer_call_fn_7056775

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
E__inference_dense_60_layer_call_and_return_conditional_losses_70549152
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
?
?
E__inference_dense_61_layer_call_and_return_conditional_losses_7054936

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
8__inference_batch_normalization_73_layer_call_fn_7056466

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
S__inference_batch_normalization_73_layer_call_and_return_conditional_losses_70543142
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
S__inference_batch_normalization_75_layer_call_and_return_conditional_losses_7056650

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
%__inference_signature_wrapper_7055546
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
"__inference__wrapped_model_70540622
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
?

?
E__inference_dense_64_layer_call_and_return_conditional_losses_7055002

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
9__inference_feed_forward_sub_net_12_layer_call_fn_7055717
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
T__inference_feed_forward_sub_net_12_layer_call_and_return_conditional_losses_70552352
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
S__inference_batch_normalization_76_layer_call_and_return_conditional_losses_7054812

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
8__inference_batch_normalization_74_layer_call_fn_7056548

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
S__inference_batch_normalization_74_layer_call_and_return_conditional_losses_70544802
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
8__inference_batch_normalization_75_layer_call_fn_7056617

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
S__inference_batch_normalization_75_layer_call_and_return_conditional_losses_70545842
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
8__inference_batch_normalization_75_layer_call_fn_7056630

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
S__inference_batch_normalization_75_layer_call_and_return_conditional_losses_70546462
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
S__inference_batch_normalization_72_layer_call_and_return_conditional_losses_7054086

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
?
?
8__inference_batch_normalization_73_layer_call_fn_7056453

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
S__inference_batch_normalization_73_layer_call_and_return_conditional_losses_70542522
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
S__inference_batch_normalization_74_layer_call_and_return_conditional_losses_7054418

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
9__inference_feed_forward_sub_net_12_layer_call_fn_7055660
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
T__inference_feed_forward_sub_net_12_layer_call_and_return_conditional_losses_70550092
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
*__inference_dense_62_layer_call_fn_7056803

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
E__inference_dense_62_layer_call_and_return_conditional_losses_70549572
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
E__inference_dense_64_layer_call_and_return_conditional_losses_7056843

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
?
?
E__inference_dense_62_layer_call_and_return_conditional_losses_7054957

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
S__inference_batch_normalization_73_layer_call_and_return_conditional_losses_7056486

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
?,
?
S__inference_batch_normalization_75_layer_call_and_return_conditional_losses_7054646

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
S__inference_batch_normalization_75_layer_call_and_return_conditional_losses_7054584

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
??
?
T__inference_feed_forward_sub_net_12_layer_call_and_return_conditional_losses_7056358
input_1L
>batch_normalization_72_assignmovingavg_readvariableop_resource:
N
@batch_normalization_72_assignmovingavg_1_readvariableop_resource:
J
<batch_normalization_72_batchnorm_mul_readvariableop_resource:
F
8batch_normalization_72_batchnorm_readvariableop_resource:
9
'dense_60_matmul_readvariableop_resource:
L
>batch_normalization_73_assignmovingavg_readvariableop_resource:N
@batch_normalization_73_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_73_batchnorm_mul_readvariableop_resource:F
8batch_normalization_73_batchnorm_readvariableop_resource:9
'dense_61_matmul_readvariableop_resource:L
>batch_normalization_74_assignmovingavg_readvariableop_resource:N
@batch_normalization_74_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_74_batchnorm_mul_readvariableop_resource:F
8batch_normalization_74_batchnorm_readvariableop_resource:9
'dense_62_matmul_readvariableop_resource:L
>batch_normalization_75_assignmovingavg_readvariableop_resource:N
@batch_normalization_75_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_75_batchnorm_mul_readvariableop_resource:F
8batch_normalization_75_batchnorm_readvariableop_resource:9
'dense_63_matmul_readvariableop_resource:L
>batch_normalization_76_assignmovingavg_readvariableop_resource:N
@batch_normalization_76_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_76_batchnorm_mul_readvariableop_resource:F
8batch_normalization_76_batchnorm_readvariableop_resource:9
'dense_64_matmul_readvariableop_resource:
6
(dense_64_biasadd_readvariableop_resource:

identity??&batch_normalization_72/AssignMovingAvg?5batch_normalization_72/AssignMovingAvg/ReadVariableOp?(batch_normalization_72/AssignMovingAvg_1?7batch_normalization_72/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_72/batchnorm/ReadVariableOp?3batch_normalization_72/batchnorm/mul/ReadVariableOp?&batch_normalization_73/AssignMovingAvg?5batch_normalization_73/AssignMovingAvg/ReadVariableOp?(batch_normalization_73/AssignMovingAvg_1?7batch_normalization_73/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_73/batchnorm/ReadVariableOp?3batch_normalization_73/batchnorm/mul/ReadVariableOp?&batch_normalization_74/AssignMovingAvg?5batch_normalization_74/AssignMovingAvg/ReadVariableOp?(batch_normalization_74/AssignMovingAvg_1?7batch_normalization_74/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_74/batchnorm/ReadVariableOp?3batch_normalization_74/batchnorm/mul/ReadVariableOp?&batch_normalization_75/AssignMovingAvg?5batch_normalization_75/AssignMovingAvg/ReadVariableOp?(batch_normalization_75/AssignMovingAvg_1?7batch_normalization_75/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_75/batchnorm/ReadVariableOp?3batch_normalization_75/batchnorm/mul/ReadVariableOp?&batch_normalization_76/AssignMovingAvg?5batch_normalization_76/AssignMovingAvg/ReadVariableOp?(batch_normalization_76/AssignMovingAvg_1?7batch_normalization_76/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_76/batchnorm/ReadVariableOp?3batch_normalization_76/batchnorm/mul/ReadVariableOp?dense_60/MatMul/ReadVariableOp?dense_61/MatMul/ReadVariableOp?dense_62/MatMul/ReadVariableOp?dense_63/MatMul/ReadVariableOp?dense_64/BiasAdd/ReadVariableOp?dense_64/MatMul/ReadVariableOp?
5batch_normalization_72/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_72/moments/mean/reduction_indices?
#batch_normalization_72/moments/meanMeaninput_1>batch_normalization_72/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2%
#batch_normalization_72/moments/mean?
+batch_normalization_72/moments/StopGradientStopGradient,batch_normalization_72/moments/mean:output:0*
T0*
_output_shapes

:
2-
+batch_normalization_72/moments/StopGradient?
0batch_normalization_72/moments/SquaredDifferenceSquaredDifferenceinput_14batch_normalization_72/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????
22
0batch_normalization_72/moments/SquaredDifference?
9batch_normalization_72/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_72/moments/variance/reduction_indices?
'batch_normalization_72/moments/varianceMean4batch_normalization_72/moments/SquaredDifference:z:0Bbatch_normalization_72/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2)
'batch_normalization_72/moments/variance?
&batch_normalization_72/moments/SqueezeSqueeze,batch_normalization_72/moments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2(
&batch_normalization_72/moments/Squeeze?
(batch_normalization_72/moments/Squeeze_1Squeeze0batch_normalization_72/moments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2*
(batch_normalization_72/moments/Squeeze_1?
,batch_normalization_72/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_72/AssignMovingAvg/decay?
+batch_normalization_72/AssignMovingAvg/CastCast5batch_normalization_72/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_72/AssignMovingAvg/Cast?
5batch_normalization_72/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_72_assignmovingavg_readvariableop_resource*
_output_shapes
:
*
dtype027
5batch_normalization_72/AssignMovingAvg/ReadVariableOp?
*batch_normalization_72/AssignMovingAvg/subSub=batch_normalization_72/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_72/moments/Squeeze:output:0*
T0*
_output_shapes
:
2,
*batch_normalization_72/AssignMovingAvg/sub?
*batch_normalization_72/AssignMovingAvg/mulMul.batch_normalization_72/AssignMovingAvg/sub:z:0/batch_normalization_72/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:
2,
*batch_normalization_72/AssignMovingAvg/mul?
&batch_normalization_72/AssignMovingAvgAssignSubVariableOp>batch_normalization_72_assignmovingavg_readvariableop_resource.batch_normalization_72/AssignMovingAvg/mul:z:06^batch_normalization_72/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_72/AssignMovingAvg?
.batch_normalization_72/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_72/AssignMovingAvg_1/decay?
-batch_normalization_72/AssignMovingAvg_1/CastCast7batch_normalization_72/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_72/AssignMovingAvg_1/Cast?
7batch_normalization_72/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_72_assignmovingavg_1_readvariableop_resource*
_output_shapes
:
*
dtype029
7batch_normalization_72/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_72/AssignMovingAvg_1/subSub?batch_normalization_72/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_72/moments/Squeeze_1:output:0*
T0*
_output_shapes
:
2.
,batch_normalization_72/AssignMovingAvg_1/sub?
,batch_normalization_72/AssignMovingAvg_1/mulMul0batch_normalization_72/AssignMovingAvg_1/sub:z:01batch_normalization_72/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:
2.
,batch_normalization_72/AssignMovingAvg_1/mul?
(batch_normalization_72/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_72_assignmovingavg_1_readvariableop_resource0batch_normalization_72/AssignMovingAvg_1/mul:z:08^batch_normalization_72/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_72/AssignMovingAvg_1?
&batch_normalization_72/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2(
&batch_normalization_72/batchnorm/add/y?
$batch_normalization_72/batchnorm/addAddV21batch_normalization_72/moments/Squeeze_1:output:0/batch_normalization_72/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2&
$batch_normalization_72/batchnorm/add?
&batch_normalization_72/batchnorm/RsqrtRsqrt(batch_normalization_72/batchnorm/add:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_72/batchnorm/Rsqrt?
3batch_normalization_72/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_72_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype025
3batch_normalization_72/batchnorm/mul/ReadVariableOp?
$batch_normalization_72/batchnorm/mulMul*batch_normalization_72/batchnorm/Rsqrt:y:0;batch_normalization_72/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2&
$batch_normalization_72/batchnorm/mul?
&batch_normalization_72/batchnorm/mul_1Mulinput_1(batch_normalization_72/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????
2(
&batch_normalization_72/batchnorm/mul_1?
&batch_normalization_72/batchnorm/mul_2Mul/batch_normalization_72/moments/Squeeze:output:0(batch_normalization_72/batchnorm/mul:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_72/batchnorm/mul_2?
/batch_normalization_72/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_72_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype021
/batch_normalization_72/batchnorm/ReadVariableOp?
$batch_normalization_72/batchnorm/subSub7batch_normalization_72/batchnorm/ReadVariableOp:value:0*batch_normalization_72/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2&
$batch_normalization_72/batchnorm/sub?
&batch_normalization_72/batchnorm/add_1AddV2*batch_normalization_72/batchnorm/mul_1:z:0(batch_normalization_72/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????
2(
&batch_normalization_72/batchnorm/add_1?
dense_60/MatMul/ReadVariableOpReadVariableOp'dense_60_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_60/MatMul/ReadVariableOp?
dense_60/MatMulMatMul*batch_normalization_72/batchnorm/add_1:z:0&dense_60/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_60/MatMul?
5batch_normalization_73/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_73/moments/mean/reduction_indices?
#batch_normalization_73/moments/meanMeandense_60/MatMul:product:0>batch_normalization_73/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_73/moments/mean?
+batch_normalization_73/moments/StopGradientStopGradient,batch_normalization_73/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_73/moments/StopGradient?
0batch_normalization_73/moments/SquaredDifferenceSquaredDifferencedense_60/MatMul:product:04batch_normalization_73/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????22
0batch_normalization_73/moments/SquaredDifference?
9batch_normalization_73/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_73/moments/variance/reduction_indices?
'batch_normalization_73/moments/varianceMean4batch_normalization_73/moments/SquaredDifference:z:0Bbatch_normalization_73/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_73/moments/variance?
&batch_normalization_73/moments/SqueezeSqueeze,batch_normalization_73/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_73/moments/Squeeze?
(batch_normalization_73/moments/Squeeze_1Squeeze0batch_normalization_73/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_73/moments/Squeeze_1?
,batch_normalization_73/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_73/AssignMovingAvg/decay?
+batch_normalization_73/AssignMovingAvg/CastCast5batch_normalization_73/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_73/AssignMovingAvg/Cast?
5batch_normalization_73/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_73_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_73/AssignMovingAvg/ReadVariableOp?
*batch_normalization_73/AssignMovingAvg/subSub=batch_normalization_73/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_73/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_73/AssignMovingAvg/sub?
*batch_normalization_73/AssignMovingAvg/mulMul.batch_normalization_73/AssignMovingAvg/sub:z:0/batch_normalization_73/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_73/AssignMovingAvg/mul?
&batch_normalization_73/AssignMovingAvgAssignSubVariableOp>batch_normalization_73_assignmovingavg_readvariableop_resource.batch_normalization_73/AssignMovingAvg/mul:z:06^batch_normalization_73/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_73/AssignMovingAvg?
.batch_normalization_73/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_73/AssignMovingAvg_1/decay?
-batch_normalization_73/AssignMovingAvg_1/CastCast7batch_normalization_73/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_73/AssignMovingAvg_1/Cast?
7batch_normalization_73/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_73_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_73/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_73/AssignMovingAvg_1/subSub?batch_normalization_73/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_73/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_73/AssignMovingAvg_1/sub?
,batch_normalization_73/AssignMovingAvg_1/mulMul0batch_normalization_73/AssignMovingAvg_1/sub:z:01batch_normalization_73/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_73/AssignMovingAvg_1/mul?
(batch_normalization_73/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_73_assignmovingavg_1_readvariableop_resource0batch_normalization_73/AssignMovingAvg_1/mul:z:08^batch_normalization_73/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_73/AssignMovingAvg_1?
&batch_normalization_73/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2(
&batch_normalization_73/batchnorm/add/y?
$batch_normalization_73/batchnorm/addAddV21batch_normalization_73/moments/Squeeze_1:output:0/batch_normalization_73/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_73/batchnorm/add?
&batch_normalization_73/batchnorm/RsqrtRsqrt(batch_normalization_73/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_73/batchnorm/Rsqrt?
3batch_normalization_73/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_73_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_73/batchnorm/mul/ReadVariableOp?
$batch_normalization_73/batchnorm/mulMul*batch_normalization_73/batchnorm/Rsqrt:y:0;batch_normalization_73/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_73/batchnorm/mul?
&batch_normalization_73/batchnorm/mul_1Muldense_60/MatMul:product:0(batch_normalization_73/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_73/batchnorm/mul_1?
&batch_normalization_73/batchnorm/mul_2Mul/batch_normalization_73/moments/Squeeze:output:0(batch_normalization_73/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_73/batchnorm/mul_2?
/batch_normalization_73/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_73_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_73/batchnorm/ReadVariableOp?
$batch_normalization_73/batchnorm/subSub7batch_normalization_73/batchnorm/ReadVariableOp:value:0*batch_normalization_73/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_73/batchnorm/sub?
&batch_normalization_73/batchnorm/add_1AddV2*batch_normalization_73/batchnorm/mul_1:z:0(batch_normalization_73/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_73/batchnorm/add_1r
ReluRelu*batch_normalization_73/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu?
dense_61/MatMul/ReadVariableOpReadVariableOp'dense_61_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_61/MatMul/ReadVariableOp?
dense_61/MatMulMatMulRelu:activations:0&dense_61/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_61/MatMul?
5batch_normalization_74/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_74/moments/mean/reduction_indices?
#batch_normalization_74/moments/meanMeandense_61/MatMul:product:0>batch_normalization_74/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_74/moments/mean?
+batch_normalization_74/moments/StopGradientStopGradient,batch_normalization_74/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_74/moments/StopGradient?
0batch_normalization_74/moments/SquaredDifferenceSquaredDifferencedense_61/MatMul:product:04batch_normalization_74/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????22
0batch_normalization_74/moments/SquaredDifference?
9batch_normalization_74/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_74/moments/variance/reduction_indices?
'batch_normalization_74/moments/varianceMean4batch_normalization_74/moments/SquaredDifference:z:0Bbatch_normalization_74/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_74/moments/variance?
&batch_normalization_74/moments/SqueezeSqueeze,batch_normalization_74/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_74/moments/Squeeze?
(batch_normalization_74/moments/Squeeze_1Squeeze0batch_normalization_74/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_74/moments/Squeeze_1?
,batch_normalization_74/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_74/AssignMovingAvg/decay?
+batch_normalization_74/AssignMovingAvg/CastCast5batch_normalization_74/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_74/AssignMovingAvg/Cast?
5batch_normalization_74/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_74_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_74/AssignMovingAvg/ReadVariableOp?
*batch_normalization_74/AssignMovingAvg/subSub=batch_normalization_74/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_74/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_74/AssignMovingAvg/sub?
*batch_normalization_74/AssignMovingAvg/mulMul.batch_normalization_74/AssignMovingAvg/sub:z:0/batch_normalization_74/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_74/AssignMovingAvg/mul?
&batch_normalization_74/AssignMovingAvgAssignSubVariableOp>batch_normalization_74_assignmovingavg_readvariableop_resource.batch_normalization_74/AssignMovingAvg/mul:z:06^batch_normalization_74/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_74/AssignMovingAvg?
.batch_normalization_74/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_74/AssignMovingAvg_1/decay?
-batch_normalization_74/AssignMovingAvg_1/CastCast7batch_normalization_74/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_74/AssignMovingAvg_1/Cast?
7batch_normalization_74/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_74_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_74/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_74/AssignMovingAvg_1/subSub?batch_normalization_74/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_74/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_74/AssignMovingAvg_1/sub?
,batch_normalization_74/AssignMovingAvg_1/mulMul0batch_normalization_74/AssignMovingAvg_1/sub:z:01batch_normalization_74/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_74/AssignMovingAvg_1/mul?
(batch_normalization_74/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_74_assignmovingavg_1_readvariableop_resource0batch_normalization_74/AssignMovingAvg_1/mul:z:08^batch_normalization_74/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_74/AssignMovingAvg_1?
&batch_normalization_74/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2(
&batch_normalization_74/batchnorm/add/y?
$batch_normalization_74/batchnorm/addAddV21batch_normalization_74/moments/Squeeze_1:output:0/batch_normalization_74/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_74/batchnorm/add?
&batch_normalization_74/batchnorm/RsqrtRsqrt(batch_normalization_74/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_74/batchnorm/Rsqrt?
3batch_normalization_74/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_74_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_74/batchnorm/mul/ReadVariableOp?
$batch_normalization_74/batchnorm/mulMul*batch_normalization_74/batchnorm/Rsqrt:y:0;batch_normalization_74/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_74/batchnorm/mul?
&batch_normalization_74/batchnorm/mul_1Muldense_61/MatMul:product:0(batch_normalization_74/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_74/batchnorm/mul_1?
&batch_normalization_74/batchnorm/mul_2Mul/batch_normalization_74/moments/Squeeze:output:0(batch_normalization_74/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_74/batchnorm/mul_2?
/batch_normalization_74/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_74_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_74/batchnorm/ReadVariableOp?
$batch_normalization_74/batchnorm/subSub7batch_normalization_74/batchnorm/ReadVariableOp:value:0*batch_normalization_74/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_74/batchnorm/sub?
&batch_normalization_74/batchnorm/add_1AddV2*batch_normalization_74/batchnorm/mul_1:z:0(batch_normalization_74/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_74/batchnorm/add_1v
Relu_1Relu*batch_normalization_74/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu_1?
dense_62/MatMul/ReadVariableOpReadVariableOp'dense_62_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_62/MatMul/ReadVariableOp?
dense_62/MatMulMatMulRelu_1:activations:0&dense_62/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_62/MatMul?
5batch_normalization_75/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_75/moments/mean/reduction_indices?
#batch_normalization_75/moments/meanMeandense_62/MatMul:product:0>batch_normalization_75/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_75/moments/mean?
+batch_normalization_75/moments/StopGradientStopGradient,batch_normalization_75/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_75/moments/StopGradient?
0batch_normalization_75/moments/SquaredDifferenceSquaredDifferencedense_62/MatMul:product:04batch_normalization_75/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????22
0batch_normalization_75/moments/SquaredDifference?
9batch_normalization_75/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_75/moments/variance/reduction_indices?
'batch_normalization_75/moments/varianceMean4batch_normalization_75/moments/SquaredDifference:z:0Bbatch_normalization_75/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_75/moments/variance?
&batch_normalization_75/moments/SqueezeSqueeze,batch_normalization_75/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_75/moments/Squeeze?
(batch_normalization_75/moments/Squeeze_1Squeeze0batch_normalization_75/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_75/moments/Squeeze_1?
,batch_normalization_75/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_75/AssignMovingAvg/decay?
+batch_normalization_75/AssignMovingAvg/CastCast5batch_normalization_75/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_75/AssignMovingAvg/Cast?
5batch_normalization_75/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_75_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_75/AssignMovingAvg/ReadVariableOp?
*batch_normalization_75/AssignMovingAvg/subSub=batch_normalization_75/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_75/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_75/AssignMovingAvg/sub?
*batch_normalization_75/AssignMovingAvg/mulMul.batch_normalization_75/AssignMovingAvg/sub:z:0/batch_normalization_75/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_75/AssignMovingAvg/mul?
&batch_normalization_75/AssignMovingAvgAssignSubVariableOp>batch_normalization_75_assignmovingavg_readvariableop_resource.batch_normalization_75/AssignMovingAvg/mul:z:06^batch_normalization_75/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_75/AssignMovingAvg?
.batch_normalization_75/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_75/AssignMovingAvg_1/decay?
-batch_normalization_75/AssignMovingAvg_1/CastCast7batch_normalization_75/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_75/AssignMovingAvg_1/Cast?
7batch_normalization_75/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_75_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_75/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_75/AssignMovingAvg_1/subSub?batch_normalization_75/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_75/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_75/AssignMovingAvg_1/sub?
,batch_normalization_75/AssignMovingAvg_1/mulMul0batch_normalization_75/AssignMovingAvg_1/sub:z:01batch_normalization_75/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_75/AssignMovingAvg_1/mul?
(batch_normalization_75/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_75_assignmovingavg_1_readvariableop_resource0batch_normalization_75/AssignMovingAvg_1/mul:z:08^batch_normalization_75/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_75/AssignMovingAvg_1?
&batch_normalization_75/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2(
&batch_normalization_75/batchnorm/add/y?
$batch_normalization_75/batchnorm/addAddV21batch_normalization_75/moments/Squeeze_1:output:0/batch_normalization_75/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_75/batchnorm/add?
&batch_normalization_75/batchnorm/RsqrtRsqrt(batch_normalization_75/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_75/batchnorm/Rsqrt?
3batch_normalization_75/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_75_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_75/batchnorm/mul/ReadVariableOp?
$batch_normalization_75/batchnorm/mulMul*batch_normalization_75/batchnorm/Rsqrt:y:0;batch_normalization_75/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_75/batchnorm/mul?
&batch_normalization_75/batchnorm/mul_1Muldense_62/MatMul:product:0(batch_normalization_75/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_75/batchnorm/mul_1?
&batch_normalization_75/batchnorm/mul_2Mul/batch_normalization_75/moments/Squeeze:output:0(batch_normalization_75/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_75/batchnorm/mul_2?
/batch_normalization_75/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_75_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_75/batchnorm/ReadVariableOp?
$batch_normalization_75/batchnorm/subSub7batch_normalization_75/batchnorm/ReadVariableOp:value:0*batch_normalization_75/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_75/batchnorm/sub?
&batch_normalization_75/batchnorm/add_1AddV2*batch_normalization_75/batchnorm/mul_1:z:0(batch_normalization_75/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_75/batchnorm/add_1v
Relu_2Relu*batch_normalization_75/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu_2?
dense_63/MatMul/ReadVariableOpReadVariableOp'dense_63_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_63/MatMul/ReadVariableOp?
dense_63/MatMulMatMulRelu_2:activations:0&dense_63/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_63/MatMul?
5batch_normalization_76/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_76/moments/mean/reduction_indices?
#batch_normalization_76/moments/meanMeandense_63/MatMul:product:0>batch_normalization_76/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_76/moments/mean?
+batch_normalization_76/moments/StopGradientStopGradient,batch_normalization_76/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_76/moments/StopGradient?
0batch_normalization_76/moments/SquaredDifferenceSquaredDifferencedense_63/MatMul:product:04batch_normalization_76/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????22
0batch_normalization_76/moments/SquaredDifference?
9batch_normalization_76/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_76/moments/variance/reduction_indices?
'batch_normalization_76/moments/varianceMean4batch_normalization_76/moments/SquaredDifference:z:0Bbatch_normalization_76/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_76/moments/variance?
&batch_normalization_76/moments/SqueezeSqueeze,batch_normalization_76/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_76/moments/Squeeze?
(batch_normalization_76/moments/Squeeze_1Squeeze0batch_normalization_76/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_76/moments/Squeeze_1?
,batch_normalization_76/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_76/AssignMovingAvg/decay?
+batch_normalization_76/AssignMovingAvg/CastCast5batch_normalization_76/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_76/AssignMovingAvg/Cast?
5batch_normalization_76/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_76_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_76/AssignMovingAvg/ReadVariableOp?
*batch_normalization_76/AssignMovingAvg/subSub=batch_normalization_76/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_76/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_76/AssignMovingAvg/sub?
*batch_normalization_76/AssignMovingAvg/mulMul.batch_normalization_76/AssignMovingAvg/sub:z:0/batch_normalization_76/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_76/AssignMovingAvg/mul?
&batch_normalization_76/AssignMovingAvgAssignSubVariableOp>batch_normalization_76_assignmovingavg_readvariableop_resource.batch_normalization_76/AssignMovingAvg/mul:z:06^batch_normalization_76/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_76/AssignMovingAvg?
.batch_normalization_76/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_76/AssignMovingAvg_1/decay?
-batch_normalization_76/AssignMovingAvg_1/CastCast7batch_normalization_76/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_76/AssignMovingAvg_1/Cast?
7batch_normalization_76/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_76_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_76/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_76/AssignMovingAvg_1/subSub?batch_normalization_76/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_76/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_76/AssignMovingAvg_1/sub?
,batch_normalization_76/AssignMovingAvg_1/mulMul0batch_normalization_76/AssignMovingAvg_1/sub:z:01batch_normalization_76/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_76/AssignMovingAvg_1/mul?
(batch_normalization_76/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_76_assignmovingavg_1_readvariableop_resource0batch_normalization_76/AssignMovingAvg_1/mul:z:08^batch_normalization_76/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_76/AssignMovingAvg_1?
&batch_normalization_76/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2(
&batch_normalization_76/batchnorm/add/y?
$batch_normalization_76/batchnorm/addAddV21batch_normalization_76/moments/Squeeze_1:output:0/batch_normalization_76/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_76/batchnorm/add?
&batch_normalization_76/batchnorm/RsqrtRsqrt(batch_normalization_76/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_76/batchnorm/Rsqrt?
3batch_normalization_76/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_76_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_76/batchnorm/mul/ReadVariableOp?
$batch_normalization_76/batchnorm/mulMul*batch_normalization_76/batchnorm/Rsqrt:y:0;batch_normalization_76/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_76/batchnorm/mul?
&batch_normalization_76/batchnorm/mul_1Muldense_63/MatMul:product:0(batch_normalization_76/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_76/batchnorm/mul_1?
&batch_normalization_76/batchnorm/mul_2Mul/batch_normalization_76/moments/Squeeze:output:0(batch_normalization_76/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_76/batchnorm/mul_2?
/batch_normalization_76/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_76_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_76/batchnorm/ReadVariableOp?
$batch_normalization_76/batchnorm/subSub7batch_normalization_76/batchnorm/ReadVariableOp:value:0*batch_normalization_76/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_76/batchnorm/sub?
&batch_normalization_76/batchnorm/add_1AddV2*batch_normalization_76/batchnorm/mul_1:z:0(batch_normalization_76/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2(
&batch_normalization_76/batchnorm/add_1v
Relu_3Relu*batch_normalization_76/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu_3?
dense_64/MatMul/ReadVariableOpReadVariableOp'dense_64_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_64/MatMul/ReadVariableOp?
dense_64/MatMulMatMulRelu_3:activations:0&dense_64/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_64/MatMul?
dense_64/BiasAdd/ReadVariableOpReadVariableOp(dense_64_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_64/BiasAdd/ReadVariableOp?
dense_64/BiasAddBiasAdddense_64/MatMul:product:0'dense_64/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_64/BiasAddt
IdentityIdentitydense_64/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp'^batch_normalization_72/AssignMovingAvg6^batch_normalization_72/AssignMovingAvg/ReadVariableOp)^batch_normalization_72/AssignMovingAvg_18^batch_normalization_72/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_72/batchnorm/ReadVariableOp4^batch_normalization_72/batchnorm/mul/ReadVariableOp'^batch_normalization_73/AssignMovingAvg6^batch_normalization_73/AssignMovingAvg/ReadVariableOp)^batch_normalization_73/AssignMovingAvg_18^batch_normalization_73/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_73/batchnorm/ReadVariableOp4^batch_normalization_73/batchnorm/mul/ReadVariableOp'^batch_normalization_74/AssignMovingAvg6^batch_normalization_74/AssignMovingAvg/ReadVariableOp)^batch_normalization_74/AssignMovingAvg_18^batch_normalization_74/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_74/batchnorm/ReadVariableOp4^batch_normalization_74/batchnorm/mul/ReadVariableOp'^batch_normalization_75/AssignMovingAvg6^batch_normalization_75/AssignMovingAvg/ReadVariableOp)^batch_normalization_75/AssignMovingAvg_18^batch_normalization_75/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_75/batchnorm/ReadVariableOp4^batch_normalization_75/batchnorm/mul/ReadVariableOp'^batch_normalization_76/AssignMovingAvg6^batch_normalization_76/AssignMovingAvg/ReadVariableOp)^batch_normalization_76/AssignMovingAvg_18^batch_normalization_76/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_76/batchnorm/ReadVariableOp4^batch_normalization_76/batchnorm/mul/ReadVariableOp^dense_60/MatMul/ReadVariableOp^dense_61/MatMul/ReadVariableOp^dense_62/MatMul/ReadVariableOp^dense_63/MatMul/ReadVariableOp ^dense_64/BiasAdd/ReadVariableOp^dense_64/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????
: : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_72/AssignMovingAvg&batch_normalization_72/AssignMovingAvg2n
5batch_normalization_72/AssignMovingAvg/ReadVariableOp5batch_normalization_72/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_72/AssignMovingAvg_1(batch_normalization_72/AssignMovingAvg_12r
7batch_normalization_72/AssignMovingAvg_1/ReadVariableOp7batch_normalization_72/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_72/batchnorm/ReadVariableOp/batch_normalization_72/batchnorm/ReadVariableOp2j
3batch_normalization_72/batchnorm/mul/ReadVariableOp3batch_normalization_72/batchnorm/mul/ReadVariableOp2P
&batch_normalization_73/AssignMovingAvg&batch_normalization_73/AssignMovingAvg2n
5batch_normalization_73/AssignMovingAvg/ReadVariableOp5batch_normalization_73/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_73/AssignMovingAvg_1(batch_normalization_73/AssignMovingAvg_12r
7batch_normalization_73/AssignMovingAvg_1/ReadVariableOp7batch_normalization_73/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_73/batchnorm/ReadVariableOp/batch_normalization_73/batchnorm/ReadVariableOp2j
3batch_normalization_73/batchnorm/mul/ReadVariableOp3batch_normalization_73/batchnorm/mul/ReadVariableOp2P
&batch_normalization_74/AssignMovingAvg&batch_normalization_74/AssignMovingAvg2n
5batch_normalization_74/AssignMovingAvg/ReadVariableOp5batch_normalization_74/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_74/AssignMovingAvg_1(batch_normalization_74/AssignMovingAvg_12r
7batch_normalization_74/AssignMovingAvg_1/ReadVariableOp7batch_normalization_74/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_74/batchnorm/ReadVariableOp/batch_normalization_74/batchnorm/ReadVariableOp2j
3batch_normalization_74/batchnorm/mul/ReadVariableOp3batch_normalization_74/batchnorm/mul/ReadVariableOp2P
&batch_normalization_75/AssignMovingAvg&batch_normalization_75/AssignMovingAvg2n
5batch_normalization_75/AssignMovingAvg/ReadVariableOp5batch_normalization_75/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_75/AssignMovingAvg_1(batch_normalization_75/AssignMovingAvg_12r
7batch_normalization_75/AssignMovingAvg_1/ReadVariableOp7batch_normalization_75/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_75/batchnorm/ReadVariableOp/batch_normalization_75/batchnorm/ReadVariableOp2j
3batch_normalization_75/batchnorm/mul/ReadVariableOp3batch_normalization_75/batchnorm/mul/ReadVariableOp2P
&batch_normalization_76/AssignMovingAvg&batch_normalization_76/AssignMovingAvg2n
5batch_normalization_76/AssignMovingAvg/ReadVariableOp5batch_normalization_76/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_76/AssignMovingAvg_1(batch_normalization_76/AssignMovingAvg_12r
7batch_normalization_76/AssignMovingAvg_1/ReadVariableOp7batch_normalization_76/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_76/batchnorm/ReadVariableOp/batch_normalization_76/batchnorm/ReadVariableOp2j
3batch_normalization_76/batchnorm/mul/ReadVariableOp3batch_normalization_76/batchnorm/mul/ReadVariableOp2@
dense_60/MatMul/ReadVariableOpdense_60/MatMul/ReadVariableOp2@
dense_61/MatMul/ReadVariableOpdense_61/MatMul/ReadVariableOp2@
dense_62/MatMul/ReadVariableOpdense_62/MatMul/ReadVariableOp2@
dense_63/MatMul/ReadVariableOpdense_63/MatMul/ReadVariableOp2B
dense_64/BiasAdd/ReadVariableOpdense_64/BiasAdd/ReadVariableOp2@
dense_64/MatMul/ReadVariableOpdense_64/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????

!
_user_specified_name	input_1"?L
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
2Fnonshared_model_1/feed_forward_sub_net_12/batch_normalization_72/gamma
S:Q
2Enonshared_model_1/feed_forward_sub_net_12/batch_normalization_72/beta
T:R2Fnonshared_model_1/feed_forward_sub_net_12/batch_normalization_73/gamma
S:Q2Enonshared_model_1/feed_forward_sub_net_12/batch_normalization_73/beta
T:R2Fnonshared_model_1/feed_forward_sub_net_12/batch_normalization_74/gamma
S:Q2Enonshared_model_1/feed_forward_sub_net_12/batch_normalization_74/beta
T:R2Fnonshared_model_1/feed_forward_sub_net_12/batch_normalization_75/gamma
S:Q2Enonshared_model_1/feed_forward_sub_net_12/batch_normalization_75/beta
T:R2Fnonshared_model_1/feed_forward_sub_net_12/batch_normalization_76/gamma
S:Q2Enonshared_model_1/feed_forward_sub_net_12/batch_normalization_76/beta
\:Z
 (2Lnonshared_model_1/feed_forward_sub_net_12/batch_normalization_72/moving_mean
`:^
 (2Pnonshared_model_1/feed_forward_sub_net_12/batch_normalization_72/moving_variance
\:Z (2Lnonshared_model_1/feed_forward_sub_net_12/batch_normalization_73/moving_mean
`:^ (2Pnonshared_model_1/feed_forward_sub_net_12/batch_normalization_73/moving_variance
\:Z (2Lnonshared_model_1/feed_forward_sub_net_12/batch_normalization_74/moving_mean
`:^ (2Pnonshared_model_1/feed_forward_sub_net_12/batch_normalization_74/moving_variance
\:Z (2Lnonshared_model_1/feed_forward_sub_net_12/batch_normalization_75/moving_mean
`:^ (2Pnonshared_model_1/feed_forward_sub_net_12/batch_normalization_75/moving_variance
\:Z (2Lnonshared_model_1/feed_forward_sub_net_12/batch_normalization_76/moving_mean
`:^ (2Pnonshared_model_1/feed_forward_sub_net_12/batch_normalization_76/moving_variance
K:I
29nonshared_model_1/feed_forward_sub_net_12/dense_60/kernel
K:I29nonshared_model_1/feed_forward_sub_net_12/dense_61/kernel
K:I29nonshared_model_1/feed_forward_sub_net_12/dense_62/kernel
K:I29nonshared_model_1/feed_forward_sub_net_12/dense_63/kernel
K:I
29nonshared_model_1/feed_forward_sub_net_12/dense_64/kernel
E:C
27nonshared_model_1/feed_forward_sub_net_12/dense_64/bias
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
"__inference__wrapped_model_7054062input_1"?
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
9__inference_feed_forward_sub_net_12_layer_call_fn_7055603
9__inference_feed_forward_sub_net_12_layer_call_fn_7055660
9__inference_feed_forward_sub_net_12_layer_call_fn_7055717
9__inference_feed_forward_sub_net_12_layer_call_fn_7055774?
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
T__inference_feed_forward_sub_net_12_layer_call_and_return_conditional_losses_7055880
T__inference_feed_forward_sub_net_12_layer_call_and_return_conditional_losses_7056066
T__inference_feed_forward_sub_net_12_layer_call_and_return_conditional_losses_7056172
T__inference_feed_forward_sub_net_12_layer_call_and_return_conditional_losses_7056358?
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
%__inference_signature_wrapper_7055546input_1"?
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
8__inference_batch_normalization_72_layer_call_fn_7056371
8__inference_batch_normalization_72_layer_call_fn_7056384?
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
S__inference_batch_normalization_72_layer_call_and_return_conditional_losses_7056404
S__inference_batch_normalization_72_layer_call_and_return_conditional_losses_7056440?
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
8__inference_batch_normalization_73_layer_call_fn_7056453
8__inference_batch_normalization_73_layer_call_fn_7056466?
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
S__inference_batch_normalization_73_layer_call_and_return_conditional_losses_7056486
S__inference_batch_normalization_73_layer_call_and_return_conditional_losses_7056522?
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
8__inference_batch_normalization_74_layer_call_fn_7056535
8__inference_batch_normalization_74_layer_call_fn_7056548?
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
S__inference_batch_normalization_74_layer_call_and_return_conditional_losses_7056568
S__inference_batch_normalization_74_layer_call_and_return_conditional_losses_7056604?
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
8__inference_batch_normalization_75_layer_call_fn_7056617
8__inference_batch_normalization_75_layer_call_fn_7056630?
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
S__inference_batch_normalization_75_layer_call_and_return_conditional_losses_7056650
S__inference_batch_normalization_75_layer_call_and_return_conditional_losses_7056686?
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
8__inference_batch_normalization_76_layer_call_fn_7056699
8__inference_batch_normalization_76_layer_call_fn_7056712?
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
S__inference_batch_normalization_76_layer_call_and_return_conditional_losses_7056732
S__inference_batch_normalization_76_layer_call_and_return_conditional_losses_7056768?
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
*__inference_dense_60_layer_call_fn_7056775?
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
E__inference_dense_60_layer_call_and_return_conditional_losses_7056782?
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
*__inference_dense_61_layer_call_fn_7056789?
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
E__inference_dense_61_layer_call_and_return_conditional_losses_7056796?
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
*__inference_dense_62_layer_call_fn_7056803?
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
E__inference_dense_62_layer_call_and_return_conditional_losses_7056810?
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
*__inference_dense_63_layer_call_fn_7056817?
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
E__inference_dense_63_layer_call_and_return_conditional_losses_7056824?
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
*__inference_dense_64_layer_call_fn_7056833?
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
E__inference_dense_64_layer_call_and_return_conditional_losses_7056843?
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
"__inference__wrapped_model_7054062?' ("!)$#*&%+,0?-
&?#
!?
input_1?????????

? "3?0
.
output_1"?
output_1?????????
?
S__inference_batch_normalization_72_layer_call_and_return_conditional_losses_7056404b3?0
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
S__inference_batch_normalization_72_layer_call_and_return_conditional_losses_7056440b3?0
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
8__inference_batch_normalization_72_layer_call_fn_7056371U3?0
)?&
 ?
inputs?????????

p 
? "??????????
?
8__inference_batch_normalization_72_layer_call_fn_7056384U3?0
)?&
 ?
inputs?????????

p
? "??????????
?
S__inference_batch_normalization_73_layer_call_and_return_conditional_losses_7056486b 3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
S__inference_batch_normalization_73_layer_call_and_return_conditional_losses_7056522b 3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? ?
8__inference_batch_normalization_73_layer_call_fn_7056453U 3?0
)?&
 ?
inputs?????????
p 
? "???????????
8__inference_batch_normalization_73_layer_call_fn_7056466U 3?0
)?&
 ?
inputs?????????
p
? "???????????
S__inference_batch_normalization_74_layer_call_and_return_conditional_losses_7056568b"!3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
S__inference_batch_normalization_74_layer_call_and_return_conditional_losses_7056604b!"3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? ?
8__inference_batch_normalization_74_layer_call_fn_7056535U"!3?0
)?&
 ?
inputs?????????
p 
? "???????????
8__inference_batch_normalization_74_layer_call_fn_7056548U!"3?0
)?&
 ?
inputs?????????
p
? "???????????
S__inference_batch_normalization_75_layer_call_and_return_conditional_losses_7056650b$#3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
S__inference_batch_normalization_75_layer_call_and_return_conditional_losses_7056686b#$3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? ?
8__inference_batch_normalization_75_layer_call_fn_7056617U$#3?0
)?&
 ?
inputs?????????
p 
? "???????????
8__inference_batch_normalization_75_layer_call_fn_7056630U#$3?0
)?&
 ?
inputs?????????
p
? "???????????
S__inference_batch_normalization_76_layer_call_and_return_conditional_losses_7056732b&%3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
S__inference_batch_normalization_76_layer_call_and_return_conditional_losses_7056768b%&3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? ?
8__inference_batch_normalization_76_layer_call_fn_7056699U&%3?0
)?&
 ?
inputs?????????
p 
? "???????????
8__inference_batch_normalization_76_layer_call_fn_7056712U%&3?0
)?&
 ?
inputs?????????
p
? "???????????
E__inference_dense_60_layer_call_and_return_conditional_losses_7056782['/?,
%?"
 ?
inputs?????????

? "%?"
?
0?????????
? |
*__inference_dense_60_layer_call_fn_7056775N'/?,
%?"
 ?
inputs?????????

? "???????????
E__inference_dense_61_layer_call_and_return_conditional_losses_7056796[(/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
*__inference_dense_61_layer_call_fn_7056789N(/?,
%?"
 ?
inputs?????????
? "???????????
E__inference_dense_62_layer_call_and_return_conditional_losses_7056810[)/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
*__inference_dense_62_layer_call_fn_7056803N)/?,
%?"
 ?
inputs?????????
? "???????????
E__inference_dense_63_layer_call_and_return_conditional_losses_7056824[*/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
*__inference_dense_63_layer_call_fn_7056817N*/?,
%?"
 ?
inputs?????????
? "???????????
E__inference_dense_64_layer_call_and_return_conditional_losses_7056843\+,/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????

? }
*__inference_dense_64_layer_call_fn_7056833O+,/?,
%?"
 ?
inputs?????????
? "??????????
?
T__inference_feed_forward_sub_net_12_layer_call_and_return_conditional_losses_7055880s' ("!)$#*&%+,.?+
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
T__inference_feed_forward_sub_net_12_layer_call_and_return_conditional_losses_7056066s' (!")#$*%&+,.?+
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
T__inference_feed_forward_sub_net_12_layer_call_and_return_conditional_losses_7056172y' ("!)$#*&%+,4?1
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
T__inference_feed_forward_sub_net_12_layer_call_and_return_conditional_losses_7056358y' (!")#$*%&+,4?1
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
9__inference_feed_forward_sub_net_12_layer_call_fn_7055603l' ("!)$#*&%+,4?1
*?'
!?
input_1?????????

p 
? "??????????
?
9__inference_feed_forward_sub_net_12_layer_call_fn_7055660f' ("!)$#*&%+,.?+
$?!
?
x?????????

p 
? "??????????
?
9__inference_feed_forward_sub_net_12_layer_call_fn_7055717f' (!")#$*%&+,.?+
$?!
?
x?????????

p
? "??????????
?
9__inference_feed_forward_sub_net_12_layer_call_fn_7055774l' (!")#$*%&+,4?1
*?'
!?
input_1?????????

p
? "??????????
?
%__inference_signature_wrapper_7055546?' ("!)$#*&%+,;?8
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