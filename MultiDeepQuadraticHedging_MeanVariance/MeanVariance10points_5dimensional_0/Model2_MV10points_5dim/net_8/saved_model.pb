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
 ?"serve*2.6.02unknown8͕
?
Gnonshared_model_1/feed_forward_sub_net_17/batch_normalization_102/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*X
shared_nameIGnonshared_model_1/feed_forward_sub_net_17/batch_normalization_102/gamma
?
[nonshared_model_1/feed_forward_sub_net_17/batch_normalization_102/gamma/Read/ReadVariableOpReadVariableOpGnonshared_model_1/feed_forward_sub_net_17/batch_normalization_102/gamma*
_output_shapes
:
*
dtype0
?
Fnonshared_model_1/feed_forward_sub_net_17/batch_normalization_102/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*W
shared_nameHFnonshared_model_1/feed_forward_sub_net_17/batch_normalization_102/beta
?
Znonshared_model_1/feed_forward_sub_net_17/batch_normalization_102/beta/Read/ReadVariableOpReadVariableOpFnonshared_model_1/feed_forward_sub_net_17/batch_normalization_102/beta*
_output_shapes
:
*
dtype0
?
Gnonshared_model_1/feed_forward_sub_net_17/batch_normalization_103/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*X
shared_nameIGnonshared_model_1/feed_forward_sub_net_17/batch_normalization_103/gamma
?
[nonshared_model_1/feed_forward_sub_net_17/batch_normalization_103/gamma/Read/ReadVariableOpReadVariableOpGnonshared_model_1/feed_forward_sub_net_17/batch_normalization_103/gamma*
_output_shapes
:*
dtype0
?
Fnonshared_model_1/feed_forward_sub_net_17/batch_normalization_103/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*W
shared_nameHFnonshared_model_1/feed_forward_sub_net_17/batch_normalization_103/beta
?
Znonshared_model_1/feed_forward_sub_net_17/batch_normalization_103/beta/Read/ReadVariableOpReadVariableOpFnonshared_model_1/feed_forward_sub_net_17/batch_normalization_103/beta*
_output_shapes
:*
dtype0
?
Gnonshared_model_1/feed_forward_sub_net_17/batch_normalization_104/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*X
shared_nameIGnonshared_model_1/feed_forward_sub_net_17/batch_normalization_104/gamma
?
[nonshared_model_1/feed_forward_sub_net_17/batch_normalization_104/gamma/Read/ReadVariableOpReadVariableOpGnonshared_model_1/feed_forward_sub_net_17/batch_normalization_104/gamma*
_output_shapes
:*
dtype0
?
Fnonshared_model_1/feed_forward_sub_net_17/batch_normalization_104/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*W
shared_nameHFnonshared_model_1/feed_forward_sub_net_17/batch_normalization_104/beta
?
Znonshared_model_1/feed_forward_sub_net_17/batch_normalization_104/beta/Read/ReadVariableOpReadVariableOpFnonshared_model_1/feed_forward_sub_net_17/batch_normalization_104/beta*
_output_shapes
:*
dtype0
?
Gnonshared_model_1/feed_forward_sub_net_17/batch_normalization_105/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*X
shared_nameIGnonshared_model_1/feed_forward_sub_net_17/batch_normalization_105/gamma
?
[nonshared_model_1/feed_forward_sub_net_17/batch_normalization_105/gamma/Read/ReadVariableOpReadVariableOpGnonshared_model_1/feed_forward_sub_net_17/batch_normalization_105/gamma*
_output_shapes
:*
dtype0
?
Fnonshared_model_1/feed_forward_sub_net_17/batch_normalization_105/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*W
shared_nameHFnonshared_model_1/feed_forward_sub_net_17/batch_normalization_105/beta
?
Znonshared_model_1/feed_forward_sub_net_17/batch_normalization_105/beta/Read/ReadVariableOpReadVariableOpFnonshared_model_1/feed_forward_sub_net_17/batch_normalization_105/beta*
_output_shapes
:*
dtype0
?
Gnonshared_model_1/feed_forward_sub_net_17/batch_normalization_106/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*X
shared_nameIGnonshared_model_1/feed_forward_sub_net_17/batch_normalization_106/gamma
?
[nonshared_model_1/feed_forward_sub_net_17/batch_normalization_106/gamma/Read/ReadVariableOpReadVariableOpGnonshared_model_1/feed_forward_sub_net_17/batch_normalization_106/gamma*
_output_shapes
:*
dtype0
?
Fnonshared_model_1/feed_forward_sub_net_17/batch_normalization_106/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*W
shared_nameHFnonshared_model_1/feed_forward_sub_net_17/batch_normalization_106/beta
?
Znonshared_model_1/feed_forward_sub_net_17/batch_normalization_106/beta/Read/ReadVariableOpReadVariableOpFnonshared_model_1/feed_forward_sub_net_17/batch_normalization_106/beta*
_output_shapes
:*
dtype0
?
Mnonshared_model_1/feed_forward_sub_net_17/batch_normalization_102/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*^
shared_nameOMnonshared_model_1/feed_forward_sub_net_17/batch_normalization_102/moving_mean
?
anonshared_model_1/feed_forward_sub_net_17/batch_normalization_102/moving_mean/Read/ReadVariableOpReadVariableOpMnonshared_model_1/feed_forward_sub_net_17/batch_normalization_102/moving_mean*
_output_shapes
:
*
dtype0
?
Qnonshared_model_1/feed_forward_sub_net_17/batch_normalization_102/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*b
shared_nameSQnonshared_model_1/feed_forward_sub_net_17/batch_normalization_102/moving_variance
?
enonshared_model_1/feed_forward_sub_net_17/batch_normalization_102/moving_variance/Read/ReadVariableOpReadVariableOpQnonshared_model_1/feed_forward_sub_net_17/batch_normalization_102/moving_variance*
_output_shapes
:
*
dtype0
?
Mnonshared_model_1/feed_forward_sub_net_17/batch_normalization_103/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*^
shared_nameOMnonshared_model_1/feed_forward_sub_net_17/batch_normalization_103/moving_mean
?
anonshared_model_1/feed_forward_sub_net_17/batch_normalization_103/moving_mean/Read/ReadVariableOpReadVariableOpMnonshared_model_1/feed_forward_sub_net_17/batch_normalization_103/moving_mean*
_output_shapes
:*
dtype0
?
Qnonshared_model_1/feed_forward_sub_net_17/batch_normalization_103/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*b
shared_nameSQnonshared_model_1/feed_forward_sub_net_17/batch_normalization_103/moving_variance
?
enonshared_model_1/feed_forward_sub_net_17/batch_normalization_103/moving_variance/Read/ReadVariableOpReadVariableOpQnonshared_model_1/feed_forward_sub_net_17/batch_normalization_103/moving_variance*
_output_shapes
:*
dtype0
?
Mnonshared_model_1/feed_forward_sub_net_17/batch_normalization_104/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*^
shared_nameOMnonshared_model_1/feed_forward_sub_net_17/batch_normalization_104/moving_mean
?
anonshared_model_1/feed_forward_sub_net_17/batch_normalization_104/moving_mean/Read/ReadVariableOpReadVariableOpMnonshared_model_1/feed_forward_sub_net_17/batch_normalization_104/moving_mean*
_output_shapes
:*
dtype0
?
Qnonshared_model_1/feed_forward_sub_net_17/batch_normalization_104/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*b
shared_nameSQnonshared_model_1/feed_forward_sub_net_17/batch_normalization_104/moving_variance
?
enonshared_model_1/feed_forward_sub_net_17/batch_normalization_104/moving_variance/Read/ReadVariableOpReadVariableOpQnonshared_model_1/feed_forward_sub_net_17/batch_normalization_104/moving_variance*
_output_shapes
:*
dtype0
?
Mnonshared_model_1/feed_forward_sub_net_17/batch_normalization_105/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*^
shared_nameOMnonshared_model_1/feed_forward_sub_net_17/batch_normalization_105/moving_mean
?
anonshared_model_1/feed_forward_sub_net_17/batch_normalization_105/moving_mean/Read/ReadVariableOpReadVariableOpMnonshared_model_1/feed_forward_sub_net_17/batch_normalization_105/moving_mean*
_output_shapes
:*
dtype0
?
Qnonshared_model_1/feed_forward_sub_net_17/batch_normalization_105/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*b
shared_nameSQnonshared_model_1/feed_forward_sub_net_17/batch_normalization_105/moving_variance
?
enonshared_model_1/feed_forward_sub_net_17/batch_normalization_105/moving_variance/Read/ReadVariableOpReadVariableOpQnonshared_model_1/feed_forward_sub_net_17/batch_normalization_105/moving_variance*
_output_shapes
:*
dtype0
?
Mnonshared_model_1/feed_forward_sub_net_17/batch_normalization_106/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*^
shared_nameOMnonshared_model_1/feed_forward_sub_net_17/batch_normalization_106/moving_mean
?
anonshared_model_1/feed_forward_sub_net_17/batch_normalization_106/moving_mean/Read/ReadVariableOpReadVariableOpMnonshared_model_1/feed_forward_sub_net_17/batch_normalization_106/moving_mean*
_output_shapes
:*
dtype0
?
Qnonshared_model_1/feed_forward_sub_net_17/batch_normalization_106/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*b
shared_nameSQnonshared_model_1/feed_forward_sub_net_17/batch_normalization_106/moving_variance
?
enonshared_model_1/feed_forward_sub_net_17/batch_normalization_106/moving_variance/Read/ReadVariableOpReadVariableOpQnonshared_model_1/feed_forward_sub_net_17/batch_normalization_106/moving_variance*
_output_shapes
:*
dtype0
?
9nonshared_model_1/feed_forward_sub_net_17/dense_85/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*J
shared_name;9nonshared_model_1/feed_forward_sub_net_17/dense_85/kernel
?
Mnonshared_model_1/feed_forward_sub_net_17/dense_85/kernel/Read/ReadVariableOpReadVariableOp9nonshared_model_1/feed_forward_sub_net_17/dense_85/kernel*
_output_shapes

:
*
dtype0
?
9nonshared_model_1/feed_forward_sub_net_17/dense_86/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*J
shared_name;9nonshared_model_1/feed_forward_sub_net_17/dense_86/kernel
?
Mnonshared_model_1/feed_forward_sub_net_17/dense_86/kernel/Read/ReadVariableOpReadVariableOp9nonshared_model_1/feed_forward_sub_net_17/dense_86/kernel*
_output_shapes

:*
dtype0
?
9nonshared_model_1/feed_forward_sub_net_17/dense_87/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*J
shared_name;9nonshared_model_1/feed_forward_sub_net_17/dense_87/kernel
?
Mnonshared_model_1/feed_forward_sub_net_17/dense_87/kernel/Read/ReadVariableOpReadVariableOp9nonshared_model_1/feed_forward_sub_net_17/dense_87/kernel*
_output_shapes

:*
dtype0
?
9nonshared_model_1/feed_forward_sub_net_17/dense_88/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*J
shared_name;9nonshared_model_1/feed_forward_sub_net_17/dense_88/kernel
?
Mnonshared_model_1/feed_forward_sub_net_17/dense_88/kernel/Read/ReadVariableOpReadVariableOp9nonshared_model_1/feed_forward_sub_net_17/dense_88/kernel*
_output_shapes

:*
dtype0
?
9nonshared_model_1/feed_forward_sub_net_17/dense_89/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*J
shared_name;9nonshared_model_1/feed_forward_sub_net_17/dense_89/kernel
?
Mnonshared_model_1/feed_forward_sub_net_17/dense_89/kernel/Read/ReadVariableOpReadVariableOp9nonshared_model_1/feed_forward_sub_net_17/dense_89/kernel*
_output_shapes

:
*
dtype0
?
7nonshared_model_1/feed_forward_sub_net_17/dense_89/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*H
shared_name97nonshared_model_1/feed_forward_sub_net_17/dense_89/bias
?
Knonshared_model_1/feed_forward_sub_net_17/dense_89/bias/Read/ReadVariableOpReadVariableOp7nonshared_model_1/feed_forward_sub_net_17/dense_89/bias*
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
VARIABLE_VALUEGnonshared_model_1/feed_forward_sub_net_17/batch_normalization_102/gamma&variables/0/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEFnonshared_model_1/feed_forward_sub_net_17/batch_normalization_102/beta&variables/1/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGnonshared_model_1/feed_forward_sub_net_17/batch_normalization_103/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEFnonshared_model_1/feed_forward_sub_net_17/batch_normalization_103/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGnonshared_model_1/feed_forward_sub_net_17/batch_normalization_104/gamma&variables/4/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEFnonshared_model_1/feed_forward_sub_net_17/batch_normalization_104/beta&variables/5/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGnonshared_model_1/feed_forward_sub_net_17/batch_normalization_105/gamma&variables/6/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEFnonshared_model_1/feed_forward_sub_net_17/batch_normalization_105/beta&variables/7/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGnonshared_model_1/feed_forward_sub_net_17/batch_normalization_106/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEFnonshared_model_1/feed_forward_sub_net_17/batch_normalization_106/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEMnonshared_model_1/feed_forward_sub_net_17/batch_normalization_102/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEQnonshared_model_1/feed_forward_sub_net_17/batch_normalization_102/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEMnonshared_model_1/feed_forward_sub_net_17/batch_normalization_103/moving_mean'variables/12/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEQnonshared_model_1/feed_forward_sub_net_17/batch_normalization_103/moving_variance'variables/13/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEMnonshared_model_1/feed_forward_sub_net_17/batch_normalization_104/moving_mean'variables/14/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEQnonshared_model_1/feed_forward_sub_net_17/batch_normalization_104/moving_variance'variables/15/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEMnonshared_model_1/feed_forward_sub_net_17/batch_normalization_105/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEQnonshared_model_1/feed_forward_sub_net_17/batch_normalization_105/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEMnonshared_model_1/feed_forward_sub_net_17/batch_normalization_106/moving_mean'variables/18/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEQnonshared_model_1/feed_forward_sub_net_17/batch_normalization_106/moving_variance'variables/19/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9nonshared_model_1/feed_forward_sub_net_17/dense_85/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9nonshared_model_1/feed_forward_sub_net_17/dense_86/kernel'variables/21/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9nonshared_model_1/feed_forward_sub_net_17/dense_87/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9nonshared_model_1/feed_forward_sub_net_17/dense_88/kernel'variables/23/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9nonshared_model_1/feed_forward_sub_net_17/dense_89/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE7nonshared_model_1/feed_forward_sub_net_17/dense_89/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1Qnonshared_model_1/feed_forward_sub_net_17/batch_normalization_102/moving_varianceGnonshared_model_1/feed_forward_sub_net_17/batch_normalization_102/gammaMnonshared_model_1/feed_forward_sub_net_17/batch_normalization_102/moving_meanFnonshared_model_1/feed_forward_sub_net_17/batch_normalization_102/beta9nonshared_model_1/feed_forward_sub_net_17/dense_85/kernelQnonshared_model_1/feed_forward_sub_net_17/batch_normalization_103/moving_varianceGnonshared_model_1/feed_forward_sub_net_17/batch_normalization_103/gammaMnonshared_model_1/feed_forward_sub_net_17/batch_normalization_103/moving_meanFnonshared_model_1/feed_forward_sub_net_17/batch_normalization_103/beta9nonshared_model_1/feed_forward_sub_net_17/dense_86/kernelQnonshared_model_1/feed_forward_sub_net_17/batch_normalization_104/moving_varianceGnonshared_model_1/feed_forward_sub_net_17/batch_normalization_104/gammaMnonshared_model_1/feed_forward_sub_net_17/batch_normalization_104/moving_meanFnonshared_model_1/feed_forward_sub_net_17/batch_normalization_104/beta9nonshared_model_1/feed_forward_sub_net_17/dense_87/kernelQnonshared_model_1/feed_forward_sub_net_17/batch_normalization_105/moving_varianceGnonshared_model_1/feed_forward_sub_net_17/batch_normalization_105/gammaMnonshared_model_1/feed_forward_sub_net_17/batch_normalization_105/moving_meanFnonshared_model_1/feed_forward_sub_net_17/batch_normalization_105/beta9nonshared_model_1/feed_forward_sub_net_17/dense_88/kernelQnonshared_model_1/feed_forward_sub_net_17/batch_normalization_106/moving_varianceGnonshared_model_1/feed_forward_sub_net_17/batch_normalization_106/gammaMnonshared_model_1/feed_forward_sub_net_17/batch_normalization_106/moving_meanFnonshared_model_1/feed_forward_sub_net_17/batch_normalization_106/beta9nonshared_model_1/feed_forward_sub_net_17/dense_89/kernel7nonshared_model_1/feed_forward_sub_net_17/dense_89/bias*&
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
%__inference_signature_wrapper_7071526
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename[nonshared_model_1/feed_forward_sub_net_17/batch_normalization_102/gamma/Read/ReadVariableOpZnonshared_model_1/feed_forward_sub_net_17/batch_normalization_102/beta/Read/ReadVariableOp[nonshared_model_1/feed_forward_sub_net_17/batch_normalization_103/gamma/Read/ReadVariableOpZnonshared_model_1/feed_forward_sub_net_17/batch_normalization_103/beta/Read/ReadVariableOp[nonshared_model_1/feed_forward_sub_net_17/batch_normalization_104/gamma/Read/ReadVariableOpZnonshared_model_1/feed_forward_sub_net_17/batch_normalization_104/beta/Read/ReadVariableOp[nonshared_model_1/feed_forward_sub_net_17/batch_normalization_105/gamma/Read/ReadVariableOpZnonshared_model_1/feed_forward_sub_net_17/batch_normalization_105/beta/Read/ReadVariableOp[nonshared_model_1/feed_forward_sub_net_17/batch_normalization_106/gamma/Read/ReadVariableOpZnonshared_model_1/feed_forward_sub_net_17/batch_normalization_106/beta/Read/ReadVariableOpanonshared_model_1/feed_forward_sub_net_17/batch_normalization_102/moving_mean/Read/ReadVariableOpenonshared_model_1/feed_forward_sub_net_17/batch_normalization_102/moving_variance/Read/ReadVariableOpanonshared_model_1/feed_forward_sub_net_17/batch_normalization_103/moving_mean/Read/ReadVariableOpenonshared_model_1/feed_forward_sub_net_17/batch_normalization_103/moving_variance/Read/ReadVariableOpanonshared_model_1/feed_forward_sub_net_17/batch_normalization_104/moving_mean/Read/ReadVariableOpenonshared_model_1/feed_forward_sub_net_17/batch_normalization_104/moving_variance/Read/ReadVariableOpanonshared_model_1/feed_forward_sub_net_17/batch_normalization_105/moving_mean/Read/ReadVariableOpenonshared_model_1/feed_forward_sub_net_17/batch_normalization_105/moving_variance/Read/ReadVariableOpanonshared_model_1/feed_forward_sub_net_17/batch_normalization_106/moving_mean/Read/ReadVariableOpenonshared_model_1/feed_forward_sub_net_17/batch_normalization_106/moving_variance/Read/ReadVariableOpMnonshared_model_1/feed_forward_sub_net_17/dense_85/kernel/Read/ReadVariableOpMnonshared_model_1/feed_forward_sub_net_17/dense_86/kernel/Read/ReadVariableOpMnonshared_model_1/feed_forward_sub_net_17/dense_87/kernel/Read/ReadVariableOpMnonshared_model_1/feed_forward_sub_net_17/dense_88/kernel/Read/ReadVariableOpMnonshared_model_1/feed_forward_sub_net_17/dense_89/kernel/Read/ReadVariableOpKnonshared_model_1/feed_forward_sub_net_17/dense_89/bias/Read/ReadVariableOpConst*'
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
 __inference__traced_save_7072924
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameGnonshared_model_1/feed_forward_sub_net_17/batch_normalization_102/gammaFnonshared_model_1/feed_forward_sub_net_17/batch_normalization_102/betaGnonshared_model_1/feed_forward_sub_net_17/batch_normalization_103/gammaFnonshared_model_1/feed_forward_sub_net_17/batch_normalization_103/betaGnonshared_model_1/feed_forward_sub_net_17/batch_normalization_104/gammaFnonshared_model_1/feed_forward_sub_net_17/batch_normalization_104/betaGnonshared_model_1/feed_forward_sub_net_17/batch_normalization_105/gammaFnonshared_model_1/feed_forward_sub_net_17/batch_normalization_105/betaGnonshared_model_1/feed_forward_sub_net_17/batch_normalization_106/gammaFnonshared_model_1/feed_forward_sub_net_17/batch_normalization_106/betaMnonshared_model_1/feed_forward_sub_net_17/batch_normalization_102/moving_meanQnonshared_model_1/feed_forward_sub_net_17/batch_normalization_102/moving_varianceMnonshared_model_1/feed_forward_sub_net_17/batch_normalization_103/moving_meanQnonshared_model_1/feed_forward_sub_net_17/batch_normalization_103/moving_varianceMnonshared_model_1/feed_forward_sub_net_17/batch_normalization_104/moving_meanQnonshared_model_1/feed_forward_sub_net_17/batch_normalization_104/moving_varianceMnonshared_model_1/feed_forward_sub_net_17/batch_normalization_105/moving_meanQnonshared_model_1/feed_forward_sub_net_17/batch_normalization_105/moving_varianceMnonshared_model_1/feed_forward_sub_net_17/batch_normalization_106/moving_meanQnonshared_model_1/feed_forward_sub_net_17/batch_normalization_106/moving_variance9nonshared_model_1/feed_forward_sub_net_17/dense_85/kernel9nonshared_model_1/feed_forward_sub_net_17/dense_86/kernel9nonshared_model_1/feed_forward_sub_net_17/dense_87/kernel9nonshared_model_1/feed_forward_sub_net_17/dense_88/kernel9nonshared_model_1/feed_forward_sub_net_17/dense_89/kernel7nonshared_model_1/feed_forward_sub_net_17/dense_89/bias*&
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
#__inference__traced_restore_7073012տ
?
?
T__inference_batch_normalization_106_layer_call_and_return_conditional_losses_7070730

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
9__inference_batch_normalization_105_layer_call_fn_7072610

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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_105_layer_call_and_return_conditional_losses_70706262
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
T__inference_feed_forward_sub_net_17_layer_call_and_return_conditional_losses_7072046
xM
?batch_normalization_102_assignmovingavg_readvariableop_resource:
O
Abatch_normalization_102_assignmovingavg_1_readvariableop_resource:
K
=batch_normalization_102_batchnorm_mul_readvariableop_resource:
G
9batch_normalization_102_batchnorm_readvariableop_resource:
9
'dense_85_matmul_readvariableop_resource:
M
?batch_normalization_103_assignmovingavg_readvariableop_resource:O
Abatch_normalization_103_assignmovingavg_1_readvariableop_resource:K
=batch_normalization_103_batchnorm_mul_readvariableop_resource:G
9batch_normalization_103_batchnorm_readvariableop_resource:9
'dense_86_matmul_readvariableop_resource:M
?batch_normalization_104_assignmovingavg_readvariableop_resource:O
Abatch_normalization_104_assignmovingavg_1_readvariableop_resource:K
=batch_normalization_104_batchnorm_mul_readvariableop_resource:G
9batch_normalization_104_batchnorm_readvariableop_resource:9
'dense_87_matmul_readvariableop_resource:M
?batch_normalization_105_assignmovingavg_readvariableop_resource:O
Abatch_normalization_105_assignmovingavg_1_readvariableop_resource:K
=batch_normalization_105_batchnorm_mul_readvariableop_resource:G
9batch_normalization_105_batchnorm_readvariableop_resource:9
'dense_88_matmul_readvariableop_resource:M
?batch_normalization_106_assignmovingavg_readvariableop_resource:O
Abatch_normalization_106_assignmovingavg_1_readvariableop_resource:K
=batch_normalization_106_batchnorm_mul_readvariableop_resource:G
9batch_normalization_106_batchnorm_readvariableop_resource:9
'dense_89_matmul_readvariableop_resource:
6
(dense_89_biasadd_readvariableop_resource:

identity??'batch_normalization_102/AssignMovingAvg?6batch_normalization_102/AssignMovingAvg/ReadVariableOp?)batch_normalization_102/AssignMovingAvg_1?8batch_normalization_102/AssignMovingAvg_1/ReadVariableOp?0batch_normalization_102/batchnorm/ReadVariableOp?4batch_normalization_102/batchnorm/mul/ReadVariableOp?'batch_normalization_103/AssignMovingAvg?6batch_normalization_103/AssignMovingAvg/ReadVariableOp?)batch_normalization_103/AssignMovingAvg_1?8batch_normalization_103/AssignMovingAvg_1/ReadVariableOp?0batch_normalization_103/batchnorm/ReadVariableOp?4batch_normalization_103/batchnorm/mul/ReadVariableOp?'batch_normalization_104/AssignMovingAvg?6batch_normalization_104/AssignMovingAvg/ReadVariableOp?)batch_normalization_104/AssignMovingAvg_1?8batch_normalization_104/AssignMovingAvg_1/ReadVariableOp?0batch_normalization_104/batchnorm/ReadVariableOp?4batch_normalization_104/batchnorm/mul/ReadVariableOp?'batch_normalization_105/AssignMovingAvg?6batch_normalization_105/AssignMovingAvg/ReadVariableOp?)batch_normalization_105/AssignMovingAvg_1?8batch_normalization_105/AssignMovingAvg_1/ReadVariableOp?0batch_normalization_105/batchnorm/ReadVariableOp?4batch_normalization_105/batchnorm/mul/ReadVariableOp?'batch_normalization_106/AssignMovingAvg?6batch_normalization_106/AssignMovingAvg/ReadVariableOp?)batch_normalization_106/AssignMovingAvg_1?8batch_normalization_106/AssignMovingAvg_1/ReadVariableOp?0batch_normalization_106/batchnorm/ReadVariableOp?4batch_normalization_106/batchnorm/mul/ReadVariableOp?dense_85/MatMul/ReadVariableOp?dense_86/MatMul/ReadVariableOp?dense_87/MatMul/ReadVariableOp?dense_88/MatMul/ReadVariableOp?dense_89/BiasAdd/ReadVariableOp?dense_89/MatMul/ReadVariableOp?
6batch_normalization_102/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 28
6batch_normalization_102/moments/mean/reduction_indices?
$batch_normalization_102/moments/meanMeanx?batch_normalization_102/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2&
$batch_normalization_102/moments/mean?
,batch_normalization_102/moments/StopGradientStopGradient-batch_normalization_102/moments/mean:output:0*
T0*
_output_shapes

:
2.
,batch_normalization_102/moments/StopGradient?
1batch_normalization_102/moments/SquaredDifferenceSquaredDifferencex5batch_normalization_102/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????
23
1batch_normalization_102/moments/SquaredDifference?
:batch_normalization_102/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2<
:batch_normalization_102/moments/variance/reduction_indices?
(batch_normalization_102/moments/varianceMean5batch_normalization_102/moments/SquaredDifference:z:0Cbatch_normalization_102/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2*
(batch_normalization_102/moments/variance?
'batch_normalization_102/moments/SqueezeSqueeze-batch_normalization_102/moments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2)
'batch_normalization_102/moments/Squeeze?
)batch_normalization_102/moments/Squeeze_1Squeeze1batch_normalization_102/moments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2+
)batch_normalization_102/moments/Squeeze_1?
-batch_normalization_102/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-batch_normalization_102/AssignMovingAvg/decay?
,batch_normalization_102/AssignMovingAvg/CastCast6batch_normalization_102/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2.
,batch_normalization_102/AssignMovingAvg/Cast?
6batch_normalization_102/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_102_assignmovingavg_readvariableop_resource*
_output_shapes
:
*
dtype028
6batch_normalization_102/AssignMovingAvg/ReadVariableOp?
+batch_normalization_102/AssignMovingAvg/subSub>batch_normalization_102/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_102/moments/Squeeze:output:0*
T0*
_output_shapes
:
2-
+batch_normalization_102/AssignMovingAvg/sub?
+batch_normalization_102/AssignMovingAvg/mulMul/batch_normalization_102/AssignMovingAvg/sub:z:00batch_normalization_102/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:
2-
+batch_normalization_102/AssignMovingAvg/mul?
'batch_normalization_102/AssignMovingAvgAssignSubVariableOp?batch_normalization_102_assignmovingavg_readvariableop_resource/batch_normalization_102/AssignMovingAvg/mul:z:07^batch_normalization_102/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_102/AssignMovingAvg?
/batch_normalization_102/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<21
/batch_normalization_102/AssignMovingAvg_1/decay?
.batch_normalization_102/AssignMovingAvg_1/CastCast8batch_normalization_102/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 20
.batch_normalization_102/AssignMovingAvg_1/Cast?
8batch_normalization_102/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_102_assignmovingavg_1_readvariableop_resource*
_output_shapes
:
*
dtype02:
8batch_normalization_102/AssignMovingAvg_1/ReadVariableOp?
-batch_normalization_102/AssignMovingAvg_1/subSub@batch_normalization_102/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_102/moments/Squeeze_1:output:0*
T0*
_output_shapes
:
2/
-batch_normalization_102/AssignMovingAvg_1/sub?
-batch_normalization_102/AssignMovingAvg_1/mulMul1batch_normalization_102/AssignMovingAvg_1/sub:z:02batch_normalization_102/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:
2/
-batch_normalization_102/AssignMovingAvg_1/mul?
)batch_normalization_102/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_102_assignmovingavg_1_readvariableop_resource1batch_normalization_102/AssignMovingAvg_1/mul:z:09^batch_normalization_102/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02+
)batch_normalization_102/AssignMovingAvg_1?
'batch_normalization_102/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2)
'batch_normalization_102/batchnorm/add/y?
%batch_normalization_102/batchnorm/addAddV22batch_normalization_102/moments/Squeeze_1:output:00batch_normalization_102/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2'
%batch_normalization_102/batchnorm/add?
'batch_normalization_102/batchnorm/RsqrtRsqrt)batch_normalization_102/batchnorm/add:z:0*
T0*
_output_shapes
:
2)
'batch_normalization_102/batchnorm/Rsqrt?
4batch_normalization_102/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_102_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype026
4batch_normalization_102/batchnorm/mul/ReadVariableOp?
%batch_normalization_102/batchnorm/mulMul+batch_normalization_102/batchnorm/Rsqrt:y:0<batch_normalization_102/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2'
%batch_normalization_102/batchnorm/mul?
'batch_normalization_102/batchnorm/mul_1Mulx)batch_normalization_102/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????
2)
'batch_normalization_102/batchnorm/mul_1?
'batch_normalization_102/batchnorm/mul_2Mul0batch_normalization_102/moments/Squeeze:output:0)batch_normalization_102/batchnorm/mul:z:0*
T0*
_output_shapes
:
2)
'batch_normalization_102/batchnorm/mul_2?
0batch_normalization_102/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_102_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype022
0batch_normalization_102/batchnorm/ReadVariableOp?
%batch_normalization_102/batchnorm/subSub8batch_normalization_102/batchnorm/ReadVariableOp:value:0+batch_normalization_102/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2'
%batch_normalization_102/batchnorm/sub?
'batch_normalization_102/batchnorm/add_1AddV2+batch_normalization_102/batchnorm/mul_1:z:0)batch_normalization_102/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????
2)
'batch_normalization_102/batchnorm/add_1?
dense_85/MatMul/ReadVariableOpReadVariableOp'dense_85_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_85/MatMul/ReadVariableOp?
dense_85/MatMulMatMul+batch_normalization_102/batchnorm/add_1:z:0&dense_85/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_85/MatMul?
6batch_normalization_103/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 28
6batch_normalization_103/moments/mean/reduction_indices?
$batch_normalization_103/moments/meanMeandense_85/MatMul:product:0?batch_normalization_103/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2&
$batch_normalization_103/moments/mean?
,batch_normalization_103/moments/StopGradientStopGradient-batch_normalization_103/moments/mean:output:0*
T0*
_output_shapes

:2.
,batch_normalization_103/moments/StopGradient?
1batch_normalization_103/moments/SquaredDifferenceSquaredDifferencedense_85/MatMul:product:05batch_normalization_103/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????23
1batch_normalization_103/moments/SquaredDifference?
:batch_normalization_103/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2<
:batch_normalization_103/moments/variance/reduction_indices?
(batch_normalization_103/moments/varianceMean5batch_normalization_103/moments/SquaredDifference:z:0Cbatch_normalization_103/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2*
(batch_normalization_103/moments/variance?
'batch_normalization_103/moments/SqueezeSqueeze-batch_normalization_103/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_103/moments/Squeeze?
)batch_normalization_103/moments/Squeeze_1Squeeze1batch_normalization_103/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2+
)batch_normalization_103/moments/Squeeze_1?
-batch_normalization_103/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-batch_normalization_103/AssignMovingAvg/decay?
,batch_normalization_103/AssignMovingAvg/CastCast6batch_normalization_103/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2.
,batch_normalization_103/AssignMovingAvg/Cast?
6batch_normalization_103/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_103_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_103/AssignMovingAvg/ReadVariableOp?
+batch_normalization_103/AssignMovingAvg/subSub>batch_normalization_103/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_103/moments/Squeeze:output:0*
T0*
_output_shapes
:2-
+batch_normalization_103/AssignMovingAvg/sub?
+batch_normalization_103/AssignMovingAvg/mulMul/batch_normalization_103/AssignMovingAvg/sub:z:00batch_normalization_103/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2-
+batch_normalization_103/AssignMovingAvg/mul?
'batch_normalization_103/AssignMovingAvgAssignSubVariableOp?batch_normalization_103_assignmovingavg_readvariableop_resource/batch_normalization_103/AssignMovingAvg/mul:z:07^batch_normalization_103/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_103/AssignMovingAvg?
/batch_normalization_103/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<21
/batch_normalization_103/AssignMovingAvg_1/decay?
.batch_normalization_103/AssignMovingAvg_1/CastCast8batch_normalization_103/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 20
.batch_normalization_103/AssignMovingAvg_1/Cast?
8batch_normalization_103/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_103_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02:
8batch_normalization_103/AssignMovingAvg_1/ReadVariableOp?
-batch_normalization_103/AssignMovingAvg_1/subSub@batch_normalization_103/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_103/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2/
-batch_normalization_103/AssignMovingAvg_1/sub?
-batch_normalization_103/AssignMovingAvg_1/mulMul1batch_normalization_103/AssignMovingAvg_1/sub:z:02batch_normalization_103/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2/
-batch_normalization_103/AssignMovingAvg_1/mul?
)batch_normalization_103/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_103_assignmovingavg_1_readvariableop_resource1batch_normalization_103/AssignMovingAvg_1/mul:z:09^batch_normalization_103/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02+
)batch_normalization_103/AssignMovingAvg_1?
'batch_normalization_103/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2)
'batch_normalization_103/batchnorm/add/y?
%batch_normalization_103/batchnorm/addAddV22batch_normalization_103/moments/Squeeze_1:output:00batch_normalization_103/batchnorm/add/y:output:0*
T0*
_output_shapes
:2'
%batch_normalization_103/batchnorm/add?
'batch_normalization_103/batchnorm/RsqrtRsqrt)batch_normalization_103/batchnorm/add:z:0*
T0*
_output_shapes
:2)
'batch_normalization_103/batchnorm/Rsqrt?
4batch_normalization_103/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_103_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype026
4batch_normalization_103/batchnorm/mul/ReadVariableOp?
%batch_normalization_103/batchnorm/mulMul+batch_normalization_103/batchnorm/Rsqrt:y:0<batch_normalization_103/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2'
%batch_normalization_103/batchnorm/mul?
'batch_normalization_103/batchnorm/mul_1Muldense_85/MatMul:product:0)batch_normalization_103/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2)
'batch_normalization_103/batchnorm/mul_1?
'batch_normalization_103/batchnorm/mul_2Mul0batch_normalization_103/moments/Squeeze:output:0)batch_normalization_103/batchnorm/mul:z:0*
T0*
_output_shapes
:2)
'batch_normalization_103/batchnorm/mul_2?
0batch_normalization_103/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_103_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype022
0batch_normalization_103/batchnorm/ReadVariableOp?
%batch_normalization_103/batchnorm/subSub8batch_normalization_103/batchnorm/ReadVariableOp:value:0+batch_normalization_103/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2'
%batch_normalization_103/batchnorm/sub?
'batch_normalization_103/batchnorm/add_1AddV2+batch_normalization_103/batchnorm/mul_1:z:0)batch_normalization_103/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2)
'batch_normalization_103/batchnorm/add_1s
ReluRelu+batch_normalization_103/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu?
dense_86/MatMul/ReadVariableOpReadVariableOp'dense_86_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_86/MatMul/ReadVariableOp?
dense_86/MatMulMatMulRelu:activations:0&dense_86/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_86/MatMul?
6batch_normalization_104/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 28
6batch_normalization_104/moments/mean/reduction_indices?
$batch_normalization_104/moments/meanMeandense_86/MatMul:product:0?batch_normalization_104/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2&
$batch_normalization_104/moments/mean?
,batch_normalization_104/moments/StopGradientStopGradient-batch_normalization_104/moments/mean:output:0*
T0*
_output_shapes

:2.
,batch_normalization_104/moments/StopGradient?
1batch_normalization_104/moments/SquaredDifferenceSquaredDifferencedense_86/MatMul:product:05batch_normalization_104/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????23
1batch_normalization_104/moments/SquaredDifference?
:batch_normalization_104/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2<
:batch_normalization_104/moments/variance/reduction_indices?
(batch_normalization_104/moments/varianceMean5batch_normalization_104/moments/SquaredDifference:z:0Cbatch_normalization_104/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2*
(batch_normalization_104/moments/variance?
'batch_normalization_104/moments/SqueezeSqueeze-batch_normalization_104/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_104/moments/Squeeze?
)batch_normalization_104/moments/Squeeze_1Squeeze1batch_normalization_104/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2+
)batch_normalization_104/moments/Squeeze_1?
-batch_normalization_104/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-batch_normalization_104/AssignMovingAvg/decay?
,batch_normalization_104/AssignMovingAvg/CastCast6batch_normalization_104/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2.
,batch_normalization_104/AssignMovingAvg/Cast?
6batch_normalization_104/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_104_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_104/AssignMovingAvg/ReadVariableOp?
+batch_normalization_104/AssignMovingAvg/subSub>batch_normalization_104/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_104/moments/Squeeze:output:0*
T0*
_output_shapes
:2-
+batch_normalization_104/AssignMovingAvg/sub?
+batch_normalization_104/AssignMovingAvg/mulMul/batch_normalization_104/AssignMovingAvg/sub:z:00batch_normalization_104/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2-
+batch_normalization_104/AssignMovingAvg/mul?
'batch_normalization_104/AssignMovingAvgAssignSubVariableOp?batch_normalization_104_assignmovingavg_readvariableop_resource/batch_normalization_104/AssignMovingAvg/mul:z:07^batch_normalization_104/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_104/AssignMovingAvg?
/batch_normalization_104/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<21
/batch_normalization_104/AssignMovingAvg_1/decay?
.batch_normalization_104/AssignMovingAvg_1/CastCast8batch_normalization_104/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 20
.batch_normalization_104/AssignMovingAvg_1/Cast?
8batch_normalization_104/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_104_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02:
8batch_normalization_104/AssignMovingAvg_1/ReadVariableOp?
-batch_normalization_104/AssignMovingAvg_1/subSub@batch_normalization_104/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_104/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2/
-batch_normalization_104/AssignMovingAvg_1/sub?
-batch_normalization_104/AssignMovingAvg_1/mulMul1batch_normalization_104/AssignMovingAvg_1/sub:z:02batch_normalization_104/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2/
-batch_normalization_104/AssignMovingAvg_1/mul?
)batch_normalization_104/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_104_assignmovingavg_1_readvariableop_resource1batch_normalization_104/AssignMovingAvg_1/mul:z:09^batch_normalization_104/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02+
)batch_normalization_104/AssignMovingAvg_1?
'batch_normalization_104/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2)
'batch_normalization_104/batchnorm/add/y?
%batch_normalization_104/batchnorm/addAddV22batch_normalization_104/moments/Squeeze_1:output:00batch_normalization_104/batchnorm/add/y:output:0*
T0*
_output_shapes
:2'
%batch_normalization_104/batchnorm/add?
'batch_normalization_104/batchnorm/RsqrtRsqrt)batch_normalization_104/batchnorm/add:z:0*
T0*
_output_shapes
:2)
'batch_normalization_104/batchnorm/Rsqrt?
4batch_normalization_104/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_104_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype026
4batch_normalization_104/batchnorm/mul/ReadVariableOp?
%batch_normalization_104/batchnorm/mulMul+batch_normalization_104/batchnorm/Rsqrt:y:0<batch_normalization_104/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2'
%batch_normalization_104/batchnorm/mul?
'batch_normalization_104/batchnorm/mul_1Muldense_86/MatMul:product:0)batch_normalization_104/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2)
'batch_normalization_104/batchnorm/mul_1?
'batch_normalization_104/batchnorm/mul_2Mul0batch_normalization_104/moments/Squeeze:output:0)batch_normalization_104/batchnorm/mul:z:0*
T0*
_output_shapes
:2)
'batch_normalization_104/batchnorm/mul_2?
0batch_normalization_104/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_104_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype022
0batch_normalization_104/batchnorm/ReadVariableOp?
%batch_normalization_104/batchnorm/subSub8batch_normalization_104/batchnorm/ReadVariableOp:value:0+batch_normalization_104/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2'
%batch_normalization_104/batchnorm/sub?
'batch_normalization_104/batchnorm/add_1AddV2+batch_normalization_104/batchnorm/mul_1:z:0)batch_normalization_104/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2)
'batch_normalization_104/batchnorm/add_1w
Relu_1Relu+batch_normalization_104/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu_1?
dense_87/MatMul/ReadVariableOpReadVariableOp'dense_87_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_87/MatMul/ReadVariableOp?
dense_87/MatMulMatMulRelu_1:activations:0&dense_87/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_87/MatMul?
6batch_normalization_105/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 28
6batch_normalization_105/moments/mean/reduction_indices?
$batch_normalization_105/moments/meanMeandense_87/MatMul:product:0?batch_normalization_105/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2&
$batch_normalization_105/moments/mean?
,batch_normalization_105/moments/StopGradientStopGradient-batch_normalization_105/moments/mean:output:0*
T0*
_output_shapes

:2.
,batch_normalization_105/moments/StopGradient?
1batch_normalization_105/moments/SquaredDifferenceSquaredDifferencedense_87/MatMul:product:05batch_normalization_105/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????23
1batch_normalization_105/moments/SquaredDifference?
:batch_normalization_105/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2<
:batch_normalization_105/moments/variance/reduction_indices?
(batch_normalization_105/moments/varianceMean5batch_normalization_105/moments/SquaredDifference:z:0Cbatch_normalization_105/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2*
(batch_normalization_105/moments/variance?
'batch_normalization_105/moments/SqueezeSqueeze-batch_normalization_105/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_105/moments/Squeeze?
)batch_normalization_105/moments/Squeeze_1Squeeze1batch_normalization_105/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2+
)batch_normalization_105/moments/Squeeze_1?
-batch_normalization_105/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-batch_normalization_105/AssignMovingAvg/decay?
,batch_normalization_105/AssignMovingAvg/CastCast6batch_normalization_105/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2.
,batch_normalization_105/AssignMovingAvg/Cast?
6batch_normalization_105/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_105_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_105/AssignMovingAvg/ReadVariableOp?
+batch_normalization_105/AssignMovingAvg/subSub>batch_normalization_105/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_105/moments/Squeeze:output:0*
T0*
_output_shapes
:2-
+batch_normalization_105/AssignMovingAvg/sub?
+batch_normalization_105/AssignMovingAvg/mulMul/batch_normalization_105/AssignMovingAvg/sub:z:00batch_normalization_105/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2-
+batch_normalization_105/AssignMovingAvg/mul?
'batch_normalization_105/AssignMovingAvgAssignSubVariableOp?batch_normalization_105_assignmovingavg_readvariableop_resource/batch_normalization_105/AssignMovingAvg/mul:z:07^batch_normalization_105/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_105/AssignMovingAvg?
/batch_normalization_105/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<21
/batch_normalization_105/AssignMovingAvg_1/decay?
.batch_normalization_105/AssignMovingAvg_1/CastCast8batch_normalization_105/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 20
.batch_normalization_105/AssignMovingAvg_1/Cast?
8batch_normalization_105/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_105_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02:
8batch_normalization_105/AssignMovingAvg_1/ReadVariableOp?
-batch_normalization_105/AssignMovingAvg_1/subSub@batch_normalization_105/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_105/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2/
-batch_normalization_105/AssignMovingAvg_1/sub?
-batch_normalization_105/AssignMovingAvg_1/mulMul1batch_normalization_105/AssignMovingAvg_1/sub:z:02batch_normalization_105/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2/
-batch_normalization_105/AssignMovingAvg_1/mul?
)batch_normalization_105/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_105_assignmovingavg_1_readvariableop_resource1batch_normalization_105/AssignMovingAvg_1/mul:z:09^batch_normalization_105/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02+
)batch_normalization_105/AssignMovingAvg_1?
'batch_normalization_105/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2)
'batch_normalization_105/batchnorm/add/y?
%batch_normalization_105/batchnorm/addAddV22batch_normalization_105/moments/Squeeze_1:output:00batch_normalization_105/batchnorm/add/y:output:0*
T0*
_output_shapes
:2'
%batch_normalization_105/batchnorm/add?
'batch_normalization_105/batchnorm/RsqrtRsqrt)batch_normalization_105/batchnorm/add:z:0*
T0*
_output_shapes
:2)
'batch_normalization_105/batchnorm/Rsqrt?
4batch_normalization_105/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_105_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype026
4batch_normalization_105/batchnorm/mul/ReadVariableOp?
%batch_normalization_105/batchnorm/mulMul+batch_normalization_105/batchnorm/Rsqrt:y:0<batch_normalization_105/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2'
%batch_normalization_105/batchnorm/mul?
'batch_normalization_105/batchnorm/mul_1Muldense_87/MatMul:product:0)batch_normalization_105/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2)
'batch_normalization_105/batchnorm/mul_1?
'batch_normalization_105/batchnorm/mul_2Mul0batch_normalization_105/moments/Squeeze:output:0)batch_normalization_105/batchnorm/mul:z:0*
T0*
_output_shapes
:2)
'batch_normalization_105/batchnorm/mul_2?
0batch_normalization_105/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_105_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype022
0batch_normalization_105/batchnorm/ReadVariableOp?
%batch_normalization_105/batchnorm/subSub8batch_normalization_105/batchnorm/ReadVariableOp:value:0+batch_normalization_105/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2'
%batch_normalization_105/batchnorm/sub?
'batch_normalization_105/batchnorm/add_1AddV2+batch_normalization_105/batchnorm/mul_1:z:0)batch_normalization_105/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2)
'batch_normalization_105/batchnorm/add_1w
Relu_2Relu+batch_normalization_105/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu_2?
dense_88/MatMul/ReadVariableOpReadVariableOp'dense_88_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_88/MatMul/ReadVariableOp?
dense_88/MatMulMatMulRelu_2:activations:0&dense_88/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_88/MatMul?
6batch_normalization_106/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 28
6batch_normalization_106/moments/mean/reduction_indices?
$batch_normalization_106/moments/meanMeandense_88/MatMul:product:0?batch_normalization_106/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2&
$batch_normalization_106/moments/mean?
,batch_normalization_106/moments/StopGradientStopGradient-batch_normalization_106/moments/mean:output:0*
T0*
_output_shapes

:2.
,batch_normalization_106/moments/StopGradient?
1batch_normalization_106/moments/SquaredDifferenceSquaredDifferencedense_88/MatMul:product:05batch_normalization_106/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????23
1batch_normalization_106/moments/SquaredDifference?
:batch_normalization_106/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2<
:batch_normalization_106/moments/variance/reduction_indices?
(batch_normalization_106/moments/varianceMean5batch_normalization_106/moments/SquaredDifference:z:0Cbatch_normalization_106/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2*
(batch_normalization_106/moments/variance?
'batch_normalization_106/moments/SqueezeSqueeze-batch_normalization_106/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_106/moments/Squeeze?
)batch_normalization_106/moments/Squeeze_1Squeeze1batch_normalization_106/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2+
)batch_normalization_106/moments/Squeeze_1?
-batch_normalization_106/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-batch_normalization_106/AssignMovingAvg/decay?
,batch_normalization_106/AssignMovingAvg/CastCast6batch_normalization_106/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2.
,batch_normalization_106/AssignMovingAvg/Cast?
6batch_normalization_106/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_106_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_106/AssignMovingAvg/ReadVariableOp?
+batch_normalization_106/AssignMovingAvg/subSub>batch_normalization_106/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_106/moments/Squeeze:output:0*
T0*
_output_shapes
:2-
+batch_normalization_106/AssignMovingAvg/sub?
+batch_normalization_106/AssignMovingAvg/mulMul/batch_normalization_106/AssignMovingAvg/sub:z:00batch_normalization_106/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2-
+batch_normalization_106/AssignMovingAvg/mul?
'batch_normalization_106/AssignMovingAvgAssignSubVariableOp?batch_normalization_106_assignmovingavg_readvariableop_resource/batch_normalization_106/AssignMovingAvg/mul:z:07^batch_normalization_106/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_106/AssignMovingAvg?
/batch_normalization_106/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<21
/batch_normalization_106/AssignMovingAvg_1/decay?
.batch_normalization_106/AssignMovingAvg_1/CastCast8batch_normalization_106/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 20
.batch_normalization_106/AssignMovingAvg_1/Cast?
8batch_normalization_106/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_106_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02:
8batch_normalization_106/AssignMovingAvg_1/ReadVariableOp?
-batch_normalization_106/AssignMovingAvg_1/subSub@batch_normalization_106/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_106/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2/
-batch_normalization_106/AssignMovingAvg_1/sub?
-batch_normalization_106/AssignMovingAvg_1/mulMul1batch_normalization_106/AssignMovingAvg_1/sub:z:02batch_normalization_106/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2/
-batch_normalization_106/AssignMovingAvg_1/mul?
)batch_normalization_106/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_106_assignmovingavg_1_readvariableop_resource1batch_normalization_106/AssignMovingAvg_1/mul:z:09^batch_normalization_106/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02+
)batch_normalization_106/AssignMovingAvg_1?
'batch_normalization_106/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2)
'batch_normalization_106/batchnorm/add/y?
%batch_normalization_106/batchnorm/addAddV22batch_normalization_106/moments/Squeeze_1:output:00batch_normalization_106/batchnorm/add/y:output:0*
T0*
_output_shapes
:2'
%batch_normalization_106/batchnorm/add?
'batch_normalization_106/batchnorm/RsqrtRsqrt)batch_normalization_106/batchnorm/add:z:0*
T0*
_output_shapes
:2)
'batch_normalization_106/batchnorm/Rsqrt?
4batch_normalization_106/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_106_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype026
4batch_normalization_106/batchnorm/mul/ReadVariableOp?
%batch_normalization_106/batchnorm/mulMul+batch_normalization_106/batchnorm/Rsqrt:y:0<batch_normalization_106/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2'
%batch_normalization_106/batchnorm/mul?
'batch_normalization_106/batchnorm/mul_1Muldense_88/MatMul:product:0)batch_normalization_106/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2)
'batch_normalization_106/batchnorm/mul_1?
'batch_normalization_106/batchnorm/mul_2Mul0batch_normalization_106/moments/Squeeze:output:0)batch_normalization_106/batchnorm/mul:z:0*
T0*
_output_shapes
:2)
'batch_normalization_106/batchnorm/mul_2?
0batch_normalization_106/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_106_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype022
0batch_normalization_106/batchnorm/ReadVariableOp?
%batch_normalization_106/batchnorm/subSub8batch_normalization_106/batchnorm/ReadVariableOp:value:0+batch_normalization_106/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2'
%batch_normalization_106/batchnorm/sub?
'batch_normalization_106/batchnorm/add_1AddV2+batch_normalization_106/batchnorm/mul_1:z:0)batch_normalization_106/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2)
'batch_normalization_106/batchnorm/add_1w
Relu_3Relu+batch_normalization_106/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu_3?
dense_89/MatMul/ReadVariableOpReadVariableOp'dense_89_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_89/MatMul/ReadVariableOp?
dense_89/MatMulMatMulRelu_3:activations:0&dense_89/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_89/MatMul?
dense_89/BiasAdd/ReadVariableOpReadVariableOp(dense_89_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_89/BiasAdd/ReadVariableOp?
dense_89/BiasAddBiasAdddense_89/MatMul:product:0'dense_89/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_89/BiasAddt
IdentityIdentitydense_89/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp(^batch_normalization_102/AssignMovingAvg7^batch_normalization_102/AssignMovingAvg/ReadVariableOp*^batch_normalization_102/AssignMovingAvg_19^batch_normalization_102/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_102/batchnorm/ReadVariableOp5^batch_normalization_102/batchnorm/mul/ReadVariableOp(^batch_normalization_103/AssignMovingAvg7^batch_normalization_103/AssignMovingAvg/ReadVariableOp*^batch_normalization_103/AssignMovingAvg_19^batch_normalization_103/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_103/batchnorm/ReadVariableOp5^batch_normalization_103/batchnorm/mul/ReadVariableOp(^batch_normalization_104/AssignMovingAvg7^batch_normalization_104/AssignMovingAvg/ReadVariableOp*^batch_normalization_104/AssignMovingAvg_19^batch_normalization_104/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_104/batchnorm/ReadVariableOp5^batch_normalization_104/batchnorm/mul/ReadVariableOp(^batch_normalization_105/AssignMovingAvg7^batch_normalization_105/AssignMovingAvg/ReadVariableOp*^batch_normalization_105/AssignMovingAvg_19^batch_normalization_105/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_105/batchnorm/ReadVariableOp5^batch_normalization_105/batchnorm/mul/ReadVariableOp(^batch_normalization_106/AssignMovingAvg7^batch_normalization_106/AssignMovingAvg/ReadVariableOp*^batch_normalization_106/AssignMovingAvg_19^batch_normalization_106/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_106/batchnorm/ReadVariableOp5^batch_normalization_106/batchnorm/mul/ReadVariableOp^dense_85/MatMul/ReadVariableOp^dense_86/MatMul/ReadVariableOp^dense_87/MatMul/ReadVariableOp^dense_88/MatMul/ReadVariableOp ^dense_89/BiasAdd/ReadVariableOp^dense_89/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????
: : : : : : : : : : : : : : : : : : : : : : : : : : 2R
'batch_normalization_102/AssignMovingAvg'batch_normalization_102/AssignMovingAvg2p
6batch_normalization_102/AssignMovingAvg/ReadVariableOp6batch_normalization_102/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_102/AssignMovingAvg_1)batch_normalization_102/AssignMovingAvg_12t
8batch_normalization_102/AssignMovingAvg_1/ReadVariableOp8batch_normalization_102/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_102/batchnorm/ReadVariableOp0batch_normalization_102/batchnorm/ReadVariableOp2l
4batch_normalization_102/batchnorm/mul/ReadVariableOp4batch_normalization_102/batchnorm/mul/ReadVariableOp2R
'batch_normalization_103/AssignMovingAvg'batch_normalization_103/AssignMovingAvg2p
6batch_normalization_103/AssignMovingAvg/ReadVariableOp6batch_normalization_103/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_103/AssignMovingAvg_1)batch_normalization_103/AssignMovingAvg_12t
8batch_normalization_103/AssignMovingAvg_1/ReadVariableOp8batch_normalization_103/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_103/batchnorm/ReadVariableOp0batch_normalization_103/batchnorm/ReadVariableOp2l
4batch_normalization_103/batchnorm/mul/ReadVariableOp4batch_normalization_103/batchnorm/mul/ReadVariableOp2R
'batch_normalization_104/AssignMovingAvg'batch_normalization_104/AssignMovingAvg2p
6batch_normalization_104/AssignMovingAvg/ReadVariableOp6batch_normalization_104/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_104/AssignMovingAvg_1)batch_normalization_104/AssignMovingAvg_12t
8batch_normalization_104/AssignMovingAvg_1/ReadVariableOp8batch_normalization_104/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_104/batchnorm/ReadVariableOp0batch_normalization_104/batchnorm/ReadVariableOp2l
4batch_normalization_104/batchnorm/mul/ReadVariableOp4batch_normalization_104/batchnorm/mul/ReadVariableOp2R
'batch_normalization_105/AssignMovingAvg'batch_normalization_105/AssignMovingAvg2p
6batch_normalization_105/AssignMovingAvg/ReadVariableOp6batch_normalization_105/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_105/AssignMovingAvg_1)batch_normalization_105/AssignMovingAvg_12t
8batch_normalization_105/AssignMovingAvg_1/ReadVariableOp8batch_normalization_105/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_105/batchnorm/ReadVariableOp0batch_normalization_105/batchnorm/ReadVariableOp2l
4batch_normalization_105/batchnorm/mul/ReadVariableOp4batch_normalization_105/batchnorm/mul/ReadVariableOp2R
'batch_normalization_106/AssignMovingAvg'batch_normalization_106/AssignMovingAvg2p
6batch_normalization_106/AssignMovingAvg/ReadVariableOp6batch_normalization_106/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_106/AssignMovingAvg_1)batch_normalization_106/AssignMovingAvg_12t
8batch_normalization_106/AssignMovingAvg_1/ReadVariableOp8batch_normalization_106/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_106/batchnorm/ReadVariableOp0batch_normalization_106/batchnorm/ReadVariableOp2l
4batch_normalization_106/batchnorm/mul/ReadVariableOp4batch_normalization_106/batchnorm/mul/ReadVariableOp2@
dense_85/MatMul/ReadVariableOpdense_85/MatMul/ReadVariableOp2@
dense_86/MatMul/ReadVariableOpdense_86/MatMul/ReadVariableOp2@
dense_87/MatMul/ReadVariableOpdense_87/MatMul/ReadVariableOp2@
dense_88/MatMul/ReadVariableOpdense_88/MatMul/ReadVariableOp2B
dense_89/BiasAdd/ReadVariableOpdense_89/BiasAdd/ReadVariableOp2@
dense_89/MatMul/ReadVariableOpdense_89/MatMul/ReadVariableOp:J F
'
_output_shapes
:?????????


_user_specified_namex
?
?
9__inference_batch_normalization_106_layer_call_fn_7072692

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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_106_layer_call_and_return_conditional_losses_70707922
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
?

?
E__inference_dense_89_layer_call_and_return_conditional_losses_7072823

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
?E
?
T__inference_feed_forward_sub_net_17_layer_call_and_return_conditional_losses_7071215
x-
batch_normalization_102_7071148:
-
batch_normalization_102_7071150:
-
batch_normalization_102_7071152:
-
batch_normalization_102_7071154:
"
dense_85_7071157:
-
batch_normalization_103_7071160:-
batch_normalization_103_7071162:-
batch_normalization_103_7071164:-
batch_normalization_103_7071166:"
dense_86_7071170:-
batch_normalization_104_7071173:-
batch_normalization_104_7071175:-
batch_normalization_104_7071177:-
batch_normalization_104_7071179:"
dense_87_7071183:-
batch_normalization_105_7071186:-
batch_normalization_105_7071188:-
batch_normalization_105_7071190:-
batch_normalization_105_7071192:"
dense_88_7071196:-
batch_normalization_106_7071199:-
batch_normalization_106_7071201:-
batch_normalization_106_7071203:-
batch_normalization_106_7071205:"
dense_89_7071209:

dense_89_7071211:

identity??/batch_normalization_102/StatefulPartitionedCall?/batch_normalization_103/StatefulPartitionedCall?/batch_normalization_104/StatefulPartitionedCall?/batch_normalization_105/StatefulPartitionedCall?/batch_normalization_106/StatefulPartitionedCall? dense_85/StatefulPartitionedCall? dense_86/StatefulPartitionedCall? dense_87/StatefulPartitionedCall? dense_88/StatefulPartitionedCall? dense_89/StatefulPartitionedCall?
/batch_normalization_102/StatefulPartitionedCallStatefulPartitionedCallxbatch_normalization_102_7071148batch_normalization_102_7071150batch_normalization_102_7071152batch_normalization_102_7071154*
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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_102_layer_call_and_return_conditional_losses_707012821
/batch_normalization_102/StatefulPartitionedCall?
 dense_85/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_102/StatefulPartitionedCall:output:0dense_85_7071157*
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
E__inference_dense_85_layer_call_and_return_conditional_losses_70708952"
 dense_85/StatefulPartitionedCall?
/batch_normalization_103/StatefulPartitionedCallStatefulPartitionedCall)dense_85/StatefulPartitionedCall:output:0batch_normalization_103_7071160batch_normalization_103_7071162batch_normalization_103_7071164batch_normalization_103_7071166*
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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_103_layer_call_and_return_conditional_losses_707029421
/batch_normalization_103/StatefulPartitionedCall?
ReluRelu8batch_normalization_103/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
Relu?
 dense_86/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0dense_86_7071170*
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
E__inference_dense_86_layer_call_and_return_conditional_losses_70709162"
 dense_86/StatefulPartitionedCall?
/batch_normalization_104/StatefulPartitionedCallStatefulPartitionedCall)dense_86/StatefulPartitionedCall:output:0batch_normalization_104_7071173batch_normalization_104_7071175batch_normalization_104_7071177batch_normalization_104_7071179*
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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_104_layer_call_and_return_conditional_losses_707046021
/batch_normalization_104/StatefulPartitionedCall?
Relu_1Relu8batch_normalization_104/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
Relu_1?
 dense_87/StatefulPartitionedCallStatefulPartitionedCallRelu_1:activations:0dense_87_7071183*
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
E__inference_dense_87_layer_call_and_return_conditional_losses_70709372"
 dense_87/StatefulPartitionedCall?
/batch_normalization_105/StatefulPartitionedCallStatefulPartitionedCall)dense_87/StatefulPartitionedCall:output:0batch_normalization_105_7071186batch_normalization_105_7071188batch_normalization_105_7071190batch_normalization_105_7071192*
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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_105_layer_call_and_return_conditional_losses_707062621
/batch_normalization_105/StatefulPartitionedCall?
Relu_2Relu8batch_normalization_105/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
Relu_2?
 dense_88/StatefulPartitionedCallStatefulPartitionedCallRelu_2:activations:0dense_88_7071196*
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
E__inference_dense_88_layer_call_and_return_conditional_losses_70709582"
 dense_88/StatefulPartitionedCall?
/batch_normalization_106/StatefulPartitionedCallStatefulPartitionedCall)dense_88/StatefulPartitionedCall:output:0batch_normalization_106_7071199batch_normalization_106_7071201batch_normalization_106_7071203batch_normalization_106_7071205*
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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_106_layer_call_and_return_conditional_losses_707079221
/batch_normalization_106/StatefulPartitionedCall?
Relu_3Relu8batch_normalization_106/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
Relu_3?
 dense_89/StatefulPartitionedCallStatefulPartitionedCallRelu_3:activations:0dense_89_7071209dense_89_7071211*
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
E__inference_dense_89_layer_call_and_return_conditional_losses_70709822"
 dense_89/StatefulPartitionedCall?
IdentityIdentity)dense_89/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp0^batch_normalization_102/StatefulPartitionedCall0^batch_normalization_103/StatefulPartitionedCall0^batch_normalization_104/StatefulPartitionedCall0^batch_normalization_105/StatefulPartitionedCall0^batch_normalization_106/StatefulPartitionedCall!^dense_85/StatefulPartitionedCall!^dense_86/StatefulPartitionedCall!^dense_87/StatefulPartitionedCall!^dense_88/StatefulPartitionedCall!^dense_89/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????
: : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_102/StatefulPartitionedCall/batch_normalization_102/StatefulPartitionedCall2b
/batch_normalization_103/StatefulPartitionedCall/batch_normalization_103/StatefulPartitionedCall2b
/batch_normalization_104/StatefulPartitionedCall/batch_normalization_104/StatefulPartitionedCall2b
/batch_normalization_105/StatefulPartitionedCall/batch_normalization_105/StatefulPartitionedCall2b
/batch_normalization_106/StatefulPartitionedCall/batch_normalization_106/StatefulPartitionedCall2D
 dense_85/StatefulPartitionedCall dense_85/StatefulPartitionedCall2D
 dense_86/StatefulPartitionedCall dense_86/StatefulPartitionedCall2D
 dense_87/StatefulPartitionedCall dense_87/StatefulPartitionedCall2D
 dense_88/StatefulPartitionedCall dense_88/StatefulPartitionedCall2D
 dense_89/StatefulPartitionedCall dense_89/StatefulPartitionedCall:J F
'
_output_shapes
:?????????


_user_specified_namex
?
?
9__inference_batch_normalization_106_layer_call_fn_7072679

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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_106_layer_call_and_return_conditional_losses_70707302
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
T__inference_batch_normalization_104_layer_call_and_return_conditional_losses_7072548

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
T__inference_batch_normalization_106_layer_call_and_return_conditional_losses_7072748

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
9__inference_batch_normalization_105_layer_call_fn_7072597

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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_105_layer_call_and_return_conditional_losses_70705642
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
؂
?
#__inference__traced_restore_7073012
file_prefixf
Xassignvariableop_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_102_gamma:
g
Yassignvariableop_1_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_102_beta:
h
Zassignvariableop_2_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_103_gamma:g
Yassignvariableop_3_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_103_beta:h
Zassignvariableop_4_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_104_gamma:g
Yassignvariableop_5_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_104_beta:h
Zassignvariableop_6_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_105_gamma:g
Yassignvariableop_7_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_105_beta:h
Zassignvariableop_8_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_106_gamma:g
Yassignvariableop_9_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_106_beta:o
aassignvariableop_10_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_102_moving_mean:
s
eassignvariableop_11_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_102_moving_variance:
o
aassignvariableop_12_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_103_moving_mean:s
eassignvariableop_13_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_103_moving_variance:o
aassignvariableop_14_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_104_moving_mean:s
eassignvariableop_15_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_104_moving_variance:o
aassignvariableop_16_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_105_moving_mean:s
eassignvariableop_17_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_105_moving_variance:o
aassignvariableop_18_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_106_moving_mean:s
eassignvariableop_19_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_106_moving_variance:_
Massignvariableop_20_nonshared_model_1_feed_forward_sub_net_17_dense_85_kernel:
_
Massignvariableop_21_nonshared_model_1_feed_forward_sub_net_17_dense_86_kernel:_
Massignvariableop_22_nonshared_model_1_feed_forward_sub_net_17_dense_87_kernel:_
Massignvariableop_23_nonshared_model_1_feed_forward_sub_net_17_dense_88_kernel:_
Massignvariableop_24_nonshared_model_1_feed_forward_sub_net_17_dense_89_kernel:
Y
Kassignvariableop_25_nonshared_model_1_feed_forward_sub_net_17_dense_89_bias:
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
AssignVariableOpAssignVariableOpXassignvariableop_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_102_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpYassignvariableop_1_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_102_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpZassignvariableop_2_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_103_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpYassignvariableop_3_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_103_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpZassignvariableop_4_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_104_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpYassignvariableop_5_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_104_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpZassignvariableop_6_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_105_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpYassignvariableop_7_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_105_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpZassignvariableop_8_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_106_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpYassignvariableop_9_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_106_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpaassignvariableop_10_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_102_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpeassignvariableop_11_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_102_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpaassignvariableop_12_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_103_moving_meanIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpeassignvariableop_13_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_103_moving_varianceIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpaassignvariableop_14_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_104_moving_meanIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpeassignvariableop_15_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_104_moving_varianceIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpaassignvariableop_16_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_105_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpeassignvariableop_17_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_105_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpaassignvariableop_18_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_106_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpeassignvariableop_19_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_106_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpMassignvariableop_20_nonshared_model_1_feed_forward_sub_net_17_dense_85_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpMassignvariableop_21_nonshared_model_1_feed_forward_sub_net_17_dense_86_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpMassignvariableop_22_nonshared_model_1_feed_forward_sub_net_17_dense_87_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpMassignvariableop_23_nonshared_model_1_feed_forward_sub_net_17_dense_88_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpMassignvariableop_24_nonshared_model_1_feed_forward_sub_net_17_dense_89_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpKassignvariableop_25_nonshared_model_1_feed_forward_sub_net_17_dense_89_biasIdentity_25:output:0"/device:CPU:0*
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
?
~
*__inference_dense_86_layer_call_fn_7072769

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
E__inference_dense_86_layer_call_and_return_conditional_losses_70709162
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
?
?
*__inference_dense_89_layer_call_fn_7072813

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
E__inference_dense_89_layer_call_and_return_conditional_losses_70709822
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
?
?
9__inference_feed_forward_sub_net_17_layer_call_fn_7071697
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
T__inference_feed_forward_sub_net_17_layer_call_and_return_conditional_losses_70712152
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
?
?
9__inference_batch_normalization_102_layer_call_fn_7072364

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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_102_layer_call_and_return_conditional_losses_70701282
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
?
?
T__inference_batch_normalization_103_layer_call_and_return_conditional_losses_7072466

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
9__inference_batch_normalization_104_layer_call_fn_7072528

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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_104_layer_call_and_return_conditional_losses_70704602
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
?
?
E__inference_dense_88_layer_call_and_return_conditional_losses_7072804

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
?
~
*__inference_dense_85_layer_call_fn_7072755

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
E__inference_dense_85_layer_call_and_return_conditional_losses_70708952
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
?
?
T__inference_batch_normalization_106_layer_call_and_return_conditional_losses_7072712

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
T__inference_batch_normalization_102_layer_call_and_return_conditional_losses_7070128

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
?
T__inference_feed_forward_sub_net_17_layer_call_and_return_conditional_losses_7071860
xG
9batch_normalization_102_batchnorm_readvariableop_resource:
K
=batch_normalization_102_batchnorm_mul_readvariableop_resource:
I
;batch_normalization_102_batchnorm_readvariableop_1_resource:
I
;batch_normalization_102_batchnorm_readvariableop_2_resource:
9
'dense_85_matmul_readvariableop_resource:
G
9batch_normalization_103_batchnorm_readvariableop_resource:K
=batch_normalization_103_batchnorm_mul_readvariableop_resource:I
;batch_normalization_103_batchnorm_readvariableop_1_resource:I
;batch_normalization_103_batchnorm_readvariableop_2_resource:9
'dense_86_matmul_readvariableop_resource:G
9batch_normalization_104_batchnorm_readvariableop_resource:K
=batch_normalization_104_batchnorm_mul_readvariableop_resource:I
;batch_normalization_104_batchnorm_readvariableop_1_resource:I
;batch_normalization_104_batchnorm_readvariableop_2_resource:9
'dense_87_matmul_readvariableop_resource:G
9batch_normalization_105_batchnorm_readvariableop_resource:K
=batch_normalization_105_batchnorm_mul_readvariableop_resource:I
;batch_normalization_105_batchnorm_readvariableop_1_resource:I
;batch_normalization_105_batchnorm_readvariableop_2_resource:9
'dense_88_matmul_readvariableop_resource:G
9batch_normalization_106_batchnorm_readvariableop_resource:K
=batch_normalization_106_batchnorm_mul_readvariableop_resource:I
;batch_normalization_106_batchnorm_readvariableop_1_resource:I
;batch_normalization_106_batchnorm_readvariableop_2_resource:9
'dense_89_matmul_readvariableop_resource:
6
(dense_89_biasadd_readvariableop_resource:

identity??0batch_normalization_102/batchnorm/ReadVariableOp?2batch_normalization_102/batchnorm/ReadVariableOp_1?2batch_normalization_102/batchnorm/ReadVariableOp_2?4batch_normalization_102/batchnorm/mul/ReadVariableOp?0batch_normalization_103/batchnorm/ReadVariableOp?2batch_normalization_103/batchnorm/ReadVariableOp_1?2batch_normalization_103/batchnorm/ReadVariableOp_2?4batch_normalization_103/batchnorm/mul/ReadVariableOp?0batch_normalization_104/batchnorm/ReadVariableOp?2batch_normalization_104/batchnorm/ReadVariableOp_1?2batch_normalization_104/batchnorm/ReadVariableOp_2?4batch_normalization_104/batchnorm/mul/ReadVariableOp?0batch_normalization_105/batchnorm/ReadVariableOp?2batch_normalization_105/batchnorm/ReadVariableOp_1?2batch_normalization_105/batchnorm/ReadVariableOp_2?4batch_normalization_105/batchnorm/mul/ReadVariableOp?0batch_normalization_106/batchnorm/ReadVariableOp?2batch_normalization_106/batchnorm/ReadVariableOp_1?2batch_normalization_106/batchnorm/ReadVariableOp_2?4batch_normalization_106/batchnorm/mul/ReadVariableOp?dense_85/MatMul/ReadVariableOp?dense_86/MatMul/ReadVariableOp?dense_87/MatMul/ReadVariableOp?dense_88/MatMul/ReadVariableOp?dense_89/BiasAdd/ReadVariableOp?dense_89/MatMul/ReadVariableOp?
0batch_normalization_102/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_102_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype022
0batch_normalization_102/batchnorm/ReadVariableOp?
'batch_normalization_102/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2)
'batch_normalization_102/batchnorm/add/y?
%batch_normalization_102/batchnorm/addAddV28batch_normalization_102/batchnorm/ReadVariableOp:value:00batch_normalization_102/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2'
%batch_normalization_102/batchnorm/add?
'batch_normalization_102/batchnorm/RsqrtRsqrt)batch_normalization_102/batchnorm/add:z:0*
T0*
_output_shapes
:
2)
'batch_normalization_102/batchnorm/Rsqrt?
4batch_normalization_102/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_102_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype026
4batch_normalization_102/batchnorm/mul/ReadVariableOp?
%batch_normalization_102/batchnorm/mulMul+batch_normalization_102/batchnorm/Rsqrt:y:0<batch_normalization_102/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2'
%batch_normalization_102/batchnorm/mul?
'batch_normalization_102/batchnorm/mul_1Mulx)batch_normalization_102/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????
2)
'batch_normalization_102/batchnorm/mul_1?
2batch_normalization_102/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_102_batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype024
2batch_normalization_102/batchnorm/ReadVariableOp_1?
'batch_normalization_102/batchnorm/mul_2Mul:batch_normalization_102/batchnorm/ReadVariableOp_1:value:0)batch_normalization_102/batchnorm/mul:z:0*
T0*
_output_shapes
:
2)
'batch_normalization_102/batchnorm/mul_2?
2batch_normalization_102/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_102_batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype024
2batch_normalization_102/batchnorm/ReadVariableOp_2?
%batch_normalization_102/batchnorm/subSub:batch_normalization_102/batchnorm/ReadVariableOp_2:value:0+batch_normalization_102/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2'
%batch_normalization_102/batchnorm/sub?
'batch_normalization_102/batchnorm/add_1AddV2+batch_normalization_102/batchnorm/mul_1:z:0)batch_normalization_102/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????
2)
'batch_normalization_102/batchnorm/add_1?
dense_85/MatMul/ReadVariableOpReadVariableOp'dense_85_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_85/MatMul/ReadVariableOp?
dense_85/MatMulMatMul+batch_normalization_102/batchnorm/add_1:z:0&dense_85/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_85/MatMul?
0batch_normalization_103/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_103_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype022
0batch_normalization_103/batchnorm/ReadVariableOp?
'batch_normalization_103/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2)
'batch_normalization_103/batchnorm/add/y?
%batch_normalization_103/batchnorm/addAddV28batch_normalization_103/batchnorm/ReadVariableOp:value:00batch_normalization_103/batchnorm/add/y:output:0*
T0*
_output_shapes
:2'
%batch_normalization_103/batchnorm/add?
'batch_normalization_103/batchnorm/RsqrtRsqrt)batch_normalization_103/batchnorm/add:z:0*
T0*
_output_shapes
:2)
'batch_normalization_103/batchnorm/Rsqrt?
4batch_normalization_103/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_103_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype026
4batch_normalization_103/batchnorm/mul/ReadVariableOp?
%batch_normalization_103/batchnorm/mulMul+batch_normalization_103/batchnorm/Rsqrt:y:0<batch_normalization_103/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2'
%batch_normalization_103/batchnorm/mul?
'batch_normalization_103/batchnorm/mul_1Muldense_85/MatMul:product:0)batch_normalization_103/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2)
'batch_normalization_103/batchnorm/mul_1?
2batch_normalization_103/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_103_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype024
2batch_normalization_103/batchnorm/ReadVariableOp_1?
'batch_normalization_103/batchnorm/mul_2Mul:batch_normalization_103/batchnorm/ReadVariableOp_1:value:0)batch_normalization_103/batchnorm/mul:z:0*
T0*
_output_shapes
:2)
'batch_normalization_103/batchnorm/mul_2?
2batch_normalization_103/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_103_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype024
2batch_normalization_103/batchnorm/ReadVariableOp_2?
%batch_normalization_103/batchnorm/subSub:batch_normalization_103/batchnorm/ReadVariableOp_2:value:0+batch_normalization_103/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2'
%batch_normalization_103/batchnorm/sub?
'batch_normalization_103/batchnorm/add_1AddV2+batch_normalization_103/batchnorm/mul_1:z:0)batch_normalization_103/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2)
'batch_normalization_103/batchnorm/add_1s
ReluRelu+batch_normalization_103/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu?
dense_86/MatMul/ReadVariableOpReadVariableOp'dense_86_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_86/MatMul/ReadVariableOp?
dense_86/MatMulMatMulRelu:activations:0&dense_86/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_86/MatMul?
0batch_normalization_104/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_104_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype022
0batch_normalization_104/batchnorm/ReadVariableOp?
'batch_normalization_104/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2)
'batch_normalization_104/batchnorm/add/y?
%batch_normalization_104/batchnorm/addAddV28batch_normalization_104/batchnorm/ReadVariableOp:value:00batch_normalization_104/batchnorm/add/y:output:0*
T0*
_output_shapes
:2'
%batch_normalization_104/batchnorm/add?
'batch_normalization_104/batchnorm/RsqrtRsqrt)batch_normalization_104/batchnorm/add:z:0*
T0*
_output_shapes
:2)
'batch_normalization_104/batchnorm/Rsqrt?
4batch_normalization_104/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_104_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype026
4batch_normalization_104/batchnorm/mul/ReadVariableOp?
%batch_normalization_104/batchnorm/mulMul+batch_normalization_104/batchnorm/Rsqrt:y:0<batch_normalization_104/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2'
%batch_normalization_104/batchnorm/mul?
'batch_normalization_104/batchnorm/mul_1Muldense_86/MatMul:product:0)batch_normalization_104/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2)
'batch_normalization_104/batchnorm/mul_1?
2batch_normalization_104/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_104_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype024
2batch_normalization_104/batchnorm/ReadVariableOp_1?
'batch_normalization_104/batchnorm/mul_2Mul:batch_normalization_104/batchnorm/ReadVariableOp_1:value:0)batch_normalization_104/batchnorm/mul:z:0*
T0*
_output_shapes
:2)
'batch_normalization_104/batchnorm/mul_2?
2batch_normalization_104/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_104_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype024
2batch_normalization_104/batchnorm/ReadVariableOp_2?
%batch_normalization_104/batchnorm/subSub:batch_normalization_104/batchnorm/ReadVariableOp_2:value:0+batch_normalization_104/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2'
%batch_normalization_104/batchnorm/sub?
'batch_normalization_104/batchnorm/add_1AddV2+batch_normalization_104/batchnorm/mul_1:z:0)batch_normalization_104/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2)
'batch_normalization_104/batchnorm/add_1w
Relu_1Relu+batch_normalization_104/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu_1?
dense_87/MatMul/ReadVariableOpReadVariableOp'dense_87_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_87/MatMul/ReadVariableOp?
dense_87/MatMulMatMulRelu_1:activations:0&dense_87/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_87/MatMul?
0batch_normalization_105/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_105_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype022
0batch_normalization_105/batchnorm/ReadVariableOp?
'batch_normalization_105/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2)
'batch_normalization_105/batchnorm/add/y?
%batch_normalization_105/batchnorm/addAddV28batch_normalization_105/batchnorm/ReadVariableOp:value:00batch_normalization_105/batchnorm/add/y:output:0*
T0*
_output_shapes
:2'
%batch_normalization_105/batchnorm/add?
'batch_normalization_105/batchnorm/RsqrtRsqrt)batch_normalization_105/batchnorm/add:z:0*
T0*
_output_shapes
:2)
'batch_normalization_105/batchnorm/Rsqrt?
4batch_normalization_105/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_105_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype026
4batch_normalization_105/batchnorm/mul/ReadVariableOp?
%batch_normalization_105/batchnorm/mulMul+batch_normalization_105/batchnorm/Rsqrt:y:0<batch_normalization_105/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2'
%batch_normalization_105/batchnorm/mul?
'batch_normalization_105/batchnorm/mul_1Muldense_87/MatMul:product:0)batch_normalization_105/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2)
'batch_normalization_105/batchnorm/mul_1?
2batch_normalization_105/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_105_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype024
2batch_normalization_105/batchnorm/ReadVariableOp_1?
'batch_normalization_105/batchnorm/mul_2Mul:batch_normalization_105/batchnorm/ReadVariableOp_1:value:0)batch_normalization_105/batchnorm/mul:z:0*
T0*
_output_shapes
:2)
'batch_normalization_105/batchnorm/mul_2?
2batch_normalization_105/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_105_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype024
2batch_normalization_105/batchnorm/ReadVariableOp_2?
%batch_normalization_105/batchnorm/subSub:batch_normalization_105/batchnorm/ReadVariableOp_2:value:0+batch_normalization_105/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2'
%batch_normalization_105/batchnorm/sub?
'batch_normalization_105/batchnorm/add_1AddV2+batch_normalization_105/batchnorm/mul_1:z:0)batch_normalization_105/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2)
'batch_normalization_105/batchnorm/add_1w
Relu_2Relu+batch_normalization_105/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu_2?
dense_88/MatMul/ReadVariableOpReadVariableOp'dense_88_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_88/MatMul/ReadVariableOp?
dense_88/MatMulMatMulRelu_2:activations:0&dense_88/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_88/MatMul?
0batch_normalization_106/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_106_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype022
0batch_normalization_106/batchnorm/ReadVariableOp?
'batch_normalization_106/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2)
'batch_normalization_106/batchnorm/add/y?
%batch_normalization_106/batchnorm/addAddV28batch_normalization_106/batchnorm/ReadVariableOp:value:00batch_normalization_106/batchnorm/add/y:output:0*
T0*
_output_shapes
:2'
%batch_normalization_106/batchnorm/add?
'batch_normalization_106/batchnorm/RsqrtRsqrt)batch_normalization_106/batchnorm/add:z:0*
T0*
_output_shapes
:2)
'batch_normalization_106/batchnorm/Rsqrt?
4batch_normalization_106/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_106_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype026
4batch_normalization_106/batchnorm/mul/ReadVariableOp?
%batch_normalization_106/batchnorm/mulMul+batch_normalization_106/batchnorm/Rsqrt:y:0<batch_normalization_106/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2'
%batch_normalization_106/batchnorm/mul?
'batch_normalization_106/batchnorm/mul_1Muldense_88/MatMul:product:0)batch_normalization_106/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2)
'batch_normalization_106/batchnorm/mul_1?
2batch_normalization_106/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_106_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype024
2batch_normalization_106/batchnorm/ReadVariableOp_1?
'batch_normalization_106/batchnorm/mul_2Mul:batch_normalization_106/batchnorm/ReadVariableOp_1:value:0)batch_normalization_106/batchnorm/mul:z:0*
T0*
_output_shapes
:2)
'batch_normalization_106/batchnorm/mul_2?
2batch_normalization_106/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_106_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype024
2batch_normalization_106/batchnorm/ReadVariableOp_2?
%batch_normalization_106/batchnorm/subSub:batch_normalization_106/batchnorm/ReadVariableOp_2:value:0+batch_normalization_106/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2'
%batch_normalization_106/batchnorm/sub?
'batch_normalization_106/batchnorm/add_1AddV2+batch_normalization_106/batchnorm/mul_1:z:0)batch_normalization_106/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2)
'batch_normalization_106/batchnorm/add_1w
Relu_3Relu+batch_normalization_106/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu_3?
dense_89/MatMul/ReadVariableOpReadVariableOp'dense_89_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_89/MatMul/ReadVariableOp?
dense_89/MatMulMatMulRelu_3:activations:0&dense_89/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_89/MatMul?
dense_89/BiasAdd/ReadVariableOpReadVariableOp(dense_89_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_89/BiasAdd/ReadVariableOp?
dense_89/BiasAddBiasAdddense_89/MatMul:product:0'dense_89/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_89/BiasAddt
IdentityIdentitydense_89/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?

NoOpNoOp1^batch_normalization_102/batchnorm/ReadVariableOp3^batch_normalization_102/batchnorm/ReadVariableOp_13^batch_normalization_102/batchnorm/ReadVariableOp_25^batch_normalization_102/batchnorm/mul/ReadVariableOp1^batch_normalization_103/batchnorm/ReadVariableOp3^batch_normalization_103/batchnorm/ReadVariableOp_13^batch_normalization_103/batchnorm/ReadVariableOp_25^batch_normalization_103/batchnorm/mul/ReadVariableOp1^batch_normalization_104/batchnorm/ReadVariableOp3^batch_normalization_104/batchnorm/ReadVariableOp_13^batch_normalization_104/batchnorm/ReadVariableOp_25^batch_normalization_104/batchnorm/mul/ReadVariableOp1^batch_normalization_105/batchnorm/ReadVariableOp3^batch_normalization_105/batchnorm/ReadVariableOp_13^batch_normalization_105/batchnorm/ReadVariableOp_25^batch_normalization_105/batchnorm/mul/ReadVariableOp1^batch_normalization_106/batchnorm/ReadVariableOp3^batch_normalization_106/batchnorm/ReadVariableOp_13^batch_normalization_106/batchnorm/ReadVariableOp_25^batch_normalization_106/batchnorm/mul/ReadVariableOp^dense_85/MatMul/ReadVariableOp^dense_86/MatMul/ReadVariableOp^dense_87/MatMul/ReadVariableOp^dense_88/MatMul/ReadVariableOp ^dense_89/BiasAdd/ReadVariableOp^dense_89/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????
: : : : : : : : : : : : : : : : : : : : : : : : : : 2d
0batch_normalization_102/batchnorm/ReadVariableOp0batch_normalization_102/batchnorm/ReadVariableOp2h
2batch_normalization_102/batchnorm/ReadVariableOp_12batch_normalization_102/batchnorm/ReadVariableOp_12h
2batch_normalization_102/batchnorm/ReadVariableOp_22batch_normalization_102/batchnorm/ReadVariableOp_22l
4batch_normalization_102/batchnorm/mul/ReadVariableOp4batch_normalization_102/batchnorm/mul/ReadVariableOp2d
0batch_normalization_103/batchnorm/ReadVariableOp0batch_normalization_103/batchnorm/ReadVariableOp2h
2batch_normalization_103/batchnorm/ReadVariableOp_12batch_normalization_103/batchnorm/ReadVariableOp_12h
2batch_normalization_103/batchnorm/ReadVariableOp_22batch_normalization_103/batchnorm/ReadVariableOp_22l
4batch_normalization_103/batchnorm/mul/ReadVariableOp4batch_normalization_103/batchnorm/mul/ReadVariableOp2d
0batch_normalization_104/batchnorm/ReadVariableOp0batch_normalization_104/batchnorm/ReadVariableOp2h
2batch_normalization_104/batchnorm/ReadVariableOp_12batch_normalization_104/batchnorm/ReadVariableOp_12h
2batch_normalization_104/batchnorm/ReadVariableOp_22batch_normalization_104/batchnorm/ReadVariableOp_22l
4batch_normalization_104/batchnorm/mul/ReadVariableOp4batch_normalization_104/batchnorm/mul/ReadVariableOp2d
0batch_normalization_105/batchnorm/ReadVariableOp0batch_normalization_105/batchnorm/ReadVariableOp2h
2batch_normalization_105/batchnorm/ReadVariableOp_12batch_normalization_105/batchnorm/ReadVariableOp_12h
2batch_normalization_105/batchnorm/ReadVariableOp_22batch_normalization_105/batchnorm/ReadVariableOp_22l
4batch_normalization_105/batchnorm/mul/ReadVariableOp4batch_normalization_105/batchnorm/mul/ReadVariableOp2d
0batch_normalization_106/batchnorm/ReadVariableOp0batch_normalization_106/batchnorm/ReadVariableOp2h
2batch_normalization_106/batchnorm/ReadVariableOp_12batch_normalization_106/batchnorm/ReadVariableOp_12h
2batch_normalization_106/batchnorm/ReadVariableOp_22batch_normalization_106/batchnorm/ReadVariableOp_22l
4batch_normalization_106/batchnorm/mul/ReadVariableOp4batch_normalization_106/batchnorm/mul/ReadVariableOp2@
dense_85/MatMul/ReadVariableOpdense_85/MatMul/ReadVariableOp2@
dense_86/MatMul/ReadVariableOpdense_86/MatMul/ReadVariableOp2@
dense_87/MatMul/ReadVariableOpdense_87/MatMul/ReadVariableOp2@
dense_88/MatMul/ReadVariableOpdense_88/MatMul/ReadVariableOp2B
dense_89/BiasAdd/ReadVariableOpdense_89/BiasAdd/ReadVariableOp2@
dense_89/MatMul/ReadVariableOpdense_89/MatMul/ReadVariableOp:J F
'
_output_shapes
:?????????


_user_specified_namex
?M
?
 __inference__traced_save_7072924
file_prefixf
bsavev2_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_102_gamma_read_readvariableope
asavev2_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_102_beta_read_readvariableopf
bsavev2_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_103_gamma_read_readvariableope
asavev2_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_103_beta_read_readvariableopf
bsavev2_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_104_gamma_read_readvariableope
asavev2_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_104_beta_read_readvariableopf
bsavev2_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_105_gamma_read_readvariableope
asavev2_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_105_beta_read_readvariableopf
bsavev2_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_106_gamma_read_readvariableope
asavev2_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_106_beta_read_readvariableopl
hsavev2_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_102_moving_mean_read_readvariableopp
lsavev2_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_102_moving_variance_read_readvariableopl
hsavev2_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_103_moving_mean_read_readvariableopp
lsavev2_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_103_moving_variance_read_readvariableopl
hsavev2_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_104_moving_mean_read_readvariableopp
lsavev2_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_104_moving_variance_read_readvariableopl
hsavev2_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_105_moving_mean_read_readvariableopp
lsavev2_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_105_moving_variance_read_readvariableopl
hsavev2_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_106_moving_mean_read_readvariableopp
lsavev2_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_106_moving_variance_read_readvariableopX
Tsavev2_nonshared_model_1_feed_forward_sub_net_17_dense_85_kernel_read_readvariableopX
Tsavev2_nonshared_model_1_feed_forward_sub_net_17_dense_86_kernel_read_readvariableopX
Tsavev2_nonshared_model_1_feed_forward_sub_net_17_dense_87_kernel_read_readvariableopX
Tsavev2_nonshared_model_1_feed_forward_sub_net_17_dense_88_kernel_read_readvariableopX
Tsavev2_nonshared_model_1_feed_forward_sub_net_17_dense_89_kernel_read_readvariableopV
Rsavev2_nonshared_model_1_feed_forward_sub_net_17_dense_89_bias_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0bsavev2_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_102_gamma_read_readvariableopasavev2_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_102_beta_read_readvariableopbsavev2_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_103_gamma_read_readvariableopasavev2_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_103_beta_read_readvariableopbsavev2_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_104_gamma_read_readvariableopasavev2_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_104_beta_read_readvariableopbsavev2_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_105_gamma_read_readvariableopasavev2_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_105_beta_read_readvariableopbsavev2_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_106_gamma_read_readvariableopasavev2_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_106_beta_read_readvariableophsavev2_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_102_moving_mean_read_readvariableoplsavev2_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_102_moving_variance_read_readvariableophsavev2_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_103_moving_mean_read_readvariableoplsavev2_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_103_moving_variance_read_readvariableophsavev2_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_104_moving_mean_read_readvariableoplsavev2_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_104_moving_variance_read_readvariableophsavev2_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_105_moving_mean_read_readvariableoplsavev2_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_105_moving_variance_read_readvariableophsavev2_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_106_moving_mean_read_readvariableoplsavev2_nonshared_model_1_feed_forward_sub_net_17_batch_normalization_106_moving_variance_read_readvariableopTsavev2_nonshared_model_1_feed_forward_sub_net_17_dense_85_kernel_read_readvariableopTsavev2_nonshared_model_1_feed_forward_sub_net_17_dense_86_kernel_read_readvariableopTsavev2_nonshared_model_1_feed_forward_sub_net_17_dense_87_kernel_read_readvariableopTsavev2_nonshared_model_1_feed_forward_sub_net_17_dense_88_kernel_read_readvariableopTsavev2_nonshared_model_1_feed_forward_sub_net_17_dense_89_kernel_read_readvariableopRsavev2_nonshared_model_1_feed_forward_sub_net_17_dense_89_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
?
?
T__inference_batch_normalization_102_layer_call_and_return_conditional_losses_7070066

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
̳
?
T__inference_feed_forward_sub_net_17_layer_call_and_return_conditional_losses_7072152
input_1G
9batch_normalization_102_batchnorm_readvariableop_resource:
K
=batch_normalization_102_batchnorm_mul_readvariableop_resource:
I
;batch_normalization_102_batchnorm_readvariableop_1_resource:
I
;batch_normalization_102_batchnorm_readvariableop_2_resource:
9
'dense_85_matmul_readvariableop_resource:
G
9batch_normalization_103_batchnorm_readvariableop_resource:K
=batch_normalization_103_batchnorm_mul_readvariableop_resource:I
;batch_normalization_103_batchnorm_readvariableop_1_resource:I
;batch_normalization_103_batchnorm_readvariableop_2_resource:9
'dense_86_matmul_readvariableop_resource:G
9batch_normalization_104_batchnorm_readvariableop_resource:K
=batch_normalization_104_batchnorm_mul_readvariableop_resource:I
;batch_normalization_104_batchnorm_readvariableop_1_resource:I
;batch_normalization_104_batchnorm_readvariableop_2_resource:9
'dense_87_matmul_readvariableop_resource:G
9batch_normalization_105_batchnorm_readvariableop_resource:K
=batch_normalization_105_batchnorm_mul_readvariableop_resource:I
;batch_normalization_105_batchnorm_readvariableop_1_resource:I
;batch_normalization_105_batchnorm_readvariableop_2_resource:9
'dense_88_matmul_readvariableop_resource:G
9batch_normalization_106_batchnorm_readvariableop_resource:K
=batch_normalization_106_batchnorm_mul_readvariableop_resource:I
;batch_normalization_106_batchnorm_readvariableop_1_resource:I
;batch_normalization_106_batchnorm_readvariableop_2_resource:9
'dense_89_matmul_readvariableop_resource:
6
(dense_89_biasadd_readvariableop_resource:

identity??0batch_normalization_102/batchnorm/ReadVariableOp?2batch_normalization_102/batchnorm/ReadVariableOp_1?2batch_normalization_102/batchnorm/ReadVariableOp_2?4batch_normalization_102/batchnorm/mul/ReadVariableOp?0batch_normalization_103/batchnorm/ReadVariableOp?2batch_normalization_103/batchnorm/ReadVariableOp_1?2batch_normalization_103/batchnorm/ReadVariableOp_2?4batch_normalization_103/batchnorm/mul/ReadVariableOp?0batch_normalization_104/batchnorm/ReadVariableOp?2batch_normalization_104/batchnorm/ReadVariableOp_1?2batch_normalization_104/batchnorm/ReadVariableOp_2?4batch_normalization_104/batchnorm/mul/ReadVariableOp?0batch_normalization_105/batchnorm/ReadVariableOp?2batch_normalization_105/batchnorm/ReadVariableOp_1?2batch_normalization_105/batchnorm/ReadVariableOp_2?4batch_normalization_105/batchnorm/mul/ReadVariableOp?0batch_normalization_106/batchnorm/ReadVariableOp?2batch_normalization_106/batchnorm/ReadVariableOp_1?2batch_normalization_106/batchnorm/ReadVariableOp_2?4batch_normalization_106/batchnorm/mul/ReadVariableOp?dense_85/MatMul/ReadVariableOp?dense_86/MatMul/ReadVariableOp?dense_87/MatMul/ReadVariableOp?dense_88/MatMul/ReadVariableOp?dense_89/BiasAdd/ReadVariableOp?dense_89/MatMul/ReadVariableOp?
0batch_normalization_102/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_102_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype022
0batch_normalization_102/batchnorm/ReadVariableOp?
'batch_normalization_102/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2)
'batch_normalization_102/batchnorm/add/y?
%batch_normalization_102/batchnorm/addAddV28batch_normalization_102/batchnorm/ReadVariableOp:value:00batch_normalization_102/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2'
%batch_normalization_102/batchnorm/add?
'batch_normalization_102/batchnorm/RsqrtRsqrt)batch_normalization_102/batchnorm/add:z:0*
T0*
_output_shapes
:
2)
'batch_normalization_102/batchnorm/Rsqrt?
4batch_normalization_102/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_102_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype026
4batch_normalization_102/batchnorm/mul/ReadVariableOp?
%batch_normalization_102/batchnorm/mulMul+batch_normalization_102/batchnorm/Rsqrt:y:0<batch_normalization_102/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2'
%batch_normalization_102/batchnorm/mul?
'batch_normalization_102/batchnorm/mul_1Mulinput_1)batch_normalization_102/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????
2)
'batch_normalization_102/batchnorm/mul_1?
2batch_normalization_102/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_102_batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype024
2batch_normalization_102/batchnorm/ReadVariableOp_1?
'batch_normalization_102/batchnorm/mul_2Mul:batch_normalization_102/batchnorm/ReadVariableOp_1:value:0)batch_normalization_102/batchnorm/mul:z:0*
T0*
_output_shapes
:
2)
'batch_normalization_102/batchnorm/mul_2?
2batch_normalization_102/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_102_batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype024
2batch_normalization_102/batchnorm/ReadVariableOp_2?
%batch_normalization_102/batchnorm/subSub:batch_normalization_102/batchnorm/ReadVariableOp_2:value:0+batch_normalization_102/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2'
%batch_normalization_102/batchnorm/sub?
'batch_normalization_102/batchnorm/add_1AddV2+batch_normalization_102/batchnorm/mul_1:z:0)batch_normalization_102/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????
2)
'batch_normalization_102/batchnorm/add_1?
dense_85/MatMul/ReadVariableOpReadVariableOp'dense_85_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_85/MatMul/ReadVariableOp?
dense_85/MatMulMatMul+batch_normalization_102/batchnorm/add_1:z:0&dense_85/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_85/MatMul?
0batch_normalization_103/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_103_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype022
0batch_normalization_103/batchnorm/ReadVariableOp?
'batch_normalization_103/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2)
'batch_normalization_103/batchnorm/add/y?
%batch_normalization_103/batchnorm/addAddV28batch_normalization_103/batchnorm/ReadVariableOp:value:00batch_normalization_103/batchnorm/add/y:output:0*
T0*
_output_shapes
:2'
%batch_normalization_103/batchnorm/add?
'batch_normalization_103/batchnorm/RsqrtRsqrt)batch_normalization_103/batchnorm/add:z:0*
T0*
_output_shapes
:2)
'batch_normalization_103/batchnorm/Rsqrt?
4batch_normalization_103/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_103_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype026
4batch_normalization_103/batchnorm/mul/ReadVariableOp?
%batch_normalization_103/batchnorm/mulMul+batch_normalization_103/batchnorm/Rsqrt:y:0<batch_normalization_103/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2'
%batch_normalization_103/batchnorm/mul?
'batch_normalization_103/batchnorm/mul_1Muldense_85/MatMul:product:0)batch_normalization_103/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2)
'batch_normalization_103/batchnorm/mul_1?
2batch_normalization_103/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_103_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype024
2batch_normalization_103/batchnorm/ReadVariableOp_1?
'batch_normalization_103/batchnorm/mul_2Mul:batch_normalization_103/batchnorm/ReadVariableOp_1:value:0)batch_normalization_103/batchnorm/mul:z:0*
T0*
_output_shapes
:2)
'batch_normalization_103/batchnorm/mul_2?
2batch_normalization_103/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_103_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype024
2batch_normalization_103/batchnorm/ReadVariableOp_2?
%batch_normalization_103/batchnorm/subSub:batch_normalization_103/batchnorm/ReadVariableOp_2:value:0+batch_normalization_103/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2'
%batch_normalization_103/batchnorm/sub?
'batch_normalization_103/batchnorm/add_1AddV2+batch_normalization_103/batchnorm/mul_1:z:0)batch_normalization_103/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2)
'batch_normalization_103/batchnorm/add_1s
ReluRelu+batch_normalization_103/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu?
dense_86/MatMul/ReadVariableOpReadVariableOp'dense_86_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_86/MatMul/ReadVariableOp?
dense_86/MatMulMatMulRelu:activations:0&dense_86/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_86/MatMul?
0batch_normalization_104/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_104_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype022
0batch_normalization_104/batchnorm/ReadVariableOp?
'batch_normalization_104/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2)
'batch_normalization_104/batchnorm/add/y?
%batch_normalization_104/batchnorm/addAddV28batch_normalization_104/batchnorm/ReadVariableOp:value:00batch_normalization_104/batchnorm/add/y:output:0*
T0*
_output_shapes
:2'
%batch_normalization_104/batchnorm/add?
'batch_normalization_104/batchnorm/RsqrtRsqrt)batch_normalization_104/batchnorm/add:z:0*
T0*
_output_shapes
:2)
'batch_normalization_104/batchnorm/Rsqrt?
4batch_normalization_104/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_104_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype026
4batch_normalization_104/batchnorm/mul/ReadVariableOp?
%batch_normalization_104/batchnorm/mulMul+batch_normalization_104/batchnorm/Rsqrt:y:0<batch_normalization_104/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2'
%batch_normalization_104/batchnorm/mul?
'batch_normalization_104/batchnorm/mul_1Muldense_86/MatMul:product:0)batch_normalization_104/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2)
'batch_normalization_104/batchnorm/mul_1?
2batch_normalization_104/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_104_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype024
2batch_normalization_104/batchnorm/ReadVariableOp_1?
'batch_normalization_104/batchnorm/mul_2Mul:batch_normalization_104/batchnorm/ReadVariableOp_1:value:0)batch_normalization_104/batchnorm/mul:z:0*
T0*
_output_shapes
:2)
'batch_normalization_104/batchnorm/mul_2?
2batch_normalization_104/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_104_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype024
2batch_normalization_104/batchnorm/ReadVariableOp_2?
%batch_normalization_104/batchnorm/subSub:batch_normalization_104/batchnorm/ReadVariableOp_2:value:0+batch_normalization_104/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2'
%batch_normalization_104/batchnorm/sub?
'batch_normalization_104/batchnorm/add_1AddV2+batch_normalization_104/batchnorm/mul_1:z:0)batch_normalization_104/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2)
'batch_normalization_104/batchnorm/add_1w
Relu_1Relu+batch_normalization_104/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu_1?
dense_87/MatMul/ReadVariableOpReadVariableOp'dense_87_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_87/MatMul/ReadVariableOp?
dense_87/MatMulMatMulRelu_1:activations:0&dense_87/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_87/MatMul?
0batch_normalization_105/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_105_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype022
0batch_normalization_105/batchnorm/ReadVariableOp?
'batch_normalization_105/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2)
'batch_normalization_105/batchnorm/add/y?
%batch_normalization_105/batchnorm/addAddV28batch_normalization_105/batchnorm/ReadVariableOp:value:00batch_normalization_105/batchnorm/add/y:output:0*
T0*
_output_shapes
:2'
%batch_normalization_105/batchnorm/add?
'batch_normalization_105/batchnorm/RsqrtRsqrt)batch_normalization_105/batchnorm/add:z:0*
T0*
_output_shapes
:2)
'batch_normalization_105/batchnorm/Rsqrt?
4batch_normalization_105/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_105_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype026
4batch_normalization_105/batchnorm/mul/ReadVariableOp?
%batch_normalization_105/batchnorm/mulMul+batch_normalization_105/batchnorm/Rsqrt:y:0<batch_normalization_105/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2'
%batch_normalization_105/batchnorm/mul?
'batch_normalization_105/batchnorm/mul_1Muldense_87/MatMul:product:0)batch_normalization_105/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2)
'batch_normalization_105/batchnorm/mul_1?
2batch_normalization_105/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_105_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype024
2batch_normalization_105/batchnorm/ReadVariableOp_1?
'batch_normalization_105/batchnorm/mul_2Mul:batch_normalization_105/batchnorm/ReadVariableOp_1:value:0)batch_normalization_105/batchnorm/mul:z:0*
T0*
_output_shapes
:2)
'batch_normalization_105/batchnorm/mul_2?
2batch_normalization_105/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_105_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype024
2batch_normalization_105/batchnorm/ReadVariableOp_2?
%batch_normalization_105/batchnorm/subSub:batch_normalization_105/batchnorm/ReadVariableOp_2:value:0+batch_normalization_105/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2'
%batch_normalization_105/batchnorm/sub?
'batch_normalization_105/batchnorm/add_1AddV2+batch_normalization_105/batchnorm/mul_1:z:0)batch_normalization_105/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2)
'batch_normalization_105/batchnorm/add_1w
Relu_2Relu+batch_normalization_105/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu_2?
dense_88/MatMul/ReadVariableOpReadVariableOp'dense_88_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_88/MatMul/ReadVariableOp?
dense_88/MatMulMatMulRelu_2:activations:0&dense_88/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_88/MatMul?
0batch_normalization_106/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_106_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype022
0batch_normalization_106/batchnorm/ReadVariableOp?
'batch_normalization_106/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2)
'batch_normalization_106/batchnorm/add/y?
%batch_normalization_106/batchnorm/addAddV28batch_normalization_106/batchnorm/ReadVariableOp:value:00batch_normalization_106/batchnorm/add/y:output:0*
T0*
_output_shapes
:2'
%batch_normalization_106/batchnorm/add?
'batch_normalization_106/batchnorm/RsqrtRsqrt)batch_normalization_106/batchnorm/add:z:0*
T0*
_output_shapes
:2)
'batch_normalization_106/batchnorm/Rsqrt?
4batch_normalization_106/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_106_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype026
4batch_normalization_106/batchnorm/mul/ReadVariableOp?
%batch_normalization_106/batchnorm/mulMul+batch_normalization_106/batchnorm/Rsqrt:y:0<batch_normalization_106/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2'
%batch_normalization_106/batchnorm/mul?
'batch_normalization_106/batchnorm/mul_1Muldense_88/MatMul:product:0)batch_normalization_106/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2)
'batch_normalization_106/batchnorm/mul_1?
2batch_normalization_106/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_106_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype024
2batch_normalization_106/batchnorm/ReadVariableOp_1?
'batch_normalization_106/batchnorm/mul_2Mul:batch_normalization_106/batchnorm/ReadVariableOp_1:value:0)batch_normalization_106/batchnorm/mul:z:0*
T0*
_output_shapes
:2)
'batch_normalization_106/batchnorm/mul_2?
2batch_normalization_106/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_106_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype024
2batch_normalization_106/batchnorm/ReadVariableOp_2?
%batch_normalization_106/batchnorm/subSub:batch_normalization_106/batchnorm/ReadVariableOp_2:value:0+batch_normalization_106/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2'
%batch_normalization_106/batchnorm/sub?
'batch_normalization_106/batchnorm/add_1AddV2+batch_normalization_106/batchnorm/mul_1:z:0)batch_normalization_106/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2)
'batch_normalization_106/batchnorm/add_1w
Relu_3Relu+batch_normalization_106/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu_3?
dense_89/MatMul/ReadVariableOpReadVariableOp'dense_89_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_89/MatMul/ReadVariableOp?
dense_89/MatMulMatMulRelu_3:activations:0&dense_89/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_89/MatMul?
dense_89/BiasAdd/ReadVariableOpReadVariableOp(dense_89_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_89/BiasAdd/ReadVariableOp?
dense_89/BiasAddBiasAdddense_89/MatMul:product:0'dense_89/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_89/BiasAddt
IdentityIdentitydense_89/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?

NoOpNoOp1^batch_normalization_102/batchnorm/ReadVariableOp3^batch_normalization_102/batchnorm/ReadVariableOp_13^batch_normalization_102/batchnorm/ReadVariableOp_25^batch_normalization_102/batchnorm/mul/ReadVariableOp1^batch_normalization_103/batchnorm/ReadVariableOp3^batch_normalization_103/batchnorm/ReadVariableOp_13^batch_normalization_103/batchnorm/ReadVariableOp_25^batch_normalization_103/batchnorm/mul/ReadVariableOp1^batch_normalization_104/batchnorm/ReadVariableOp3^batch_normalization_104/batchnorm/ReadVariableOp_13^batch_normalization_104/batchnorm/ReadVariableOp_25^batch_normalization_104/batchnorm/mul/ReadVariableOp1^batch_normalization_105/batchnorm/ReadVariableOp3^batch_normalization_105/batchnorm/ReadVariableOp_13^batch_normalization_105/batchnorm/ReadVariableOp_25^batch_normalization_105/batchnorm/mul/ReadVariableOp1^batch_normalization_106/batchnorm/ReadVariableOp3^batch_normalization_106/batchnorm/ReadVariableOp_13^batch_normalization_106/batchnorm/ReadVariableOp_25^batch_normalization_106/batchnorm/mul/ReadVariableOp^dense_85/MatMul/ReadVariableOp^dense_86/MatMul/ReadVariableOp^dense_87/MatMul/ReadVariableOp^dense_88/MatMul/ReadVariableOp ^dense_89/BiasAdd/ReadVariableOp^dense_89/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????
: : : : : : : : : : : : : : : : : : : : : : : : : : 2d
0batch_normalization_102/batchnorm/ReadVariableOp0batch_normalization_102/batchnorm/ReadVariableOp2h
2batch_normalization_102/batchnorm/ReadVariableOp_12batch_normalization_102/batchnorm/ReadVariableOp_12h
2batch_normalization_102/batchnorm/ReadVariableOp_22batch_normalization_102/batchnorm/ReadVariableOp_22l
4batch_normalization_102/batchnorm/mul/ReadVariableOp4batch_normalization_102/batchnorm/mul/ReadVariableOp2d
0batch_normalization_103/batchnorm/ReadVariableOp0batch_normalization_103/batchnorm/ReadVariableOp2h
2batch_normalization_103/batchnorm/ReadVariableOp_12batch_normalization_103/batchnorm/ReadVariableOp_12h
2batch_normalization_103/batchnorm/ReadVariableOp_22batch_normalization_103/batchnorm/ReadVariableOp_22l
4batch_normalization_103/batchnorm/mul/ReadVariableOp4batch_normalization_103/batchnorm/mul/ReadVariableOp2d
0batch_normalization_104/batchnorm/ReadVariableOp0batch_normalization_104/batchnorm/ReadVariableOp2h
2batch_normalization_104/batchnorm/ReadVariableOp_12batch_normalization_104/batchnorm/ReadVariableOp_12h
2batch_normalization_104/batchnorm/ReadVariableOp_22batch_normalization_104/batchnorm/ReadVariableOp_22l
4batch_normalization_104/batchnorm/mul/ReadVariableOp4batch_normalization_104/batchnorm/mul/ReadVariableOp2d
0batch_normalization_105/batchnorm/ReadVariableOp0batch_normalization_105/batchnorm/ReadVariableOp2h
2batch_normalization_105/batchnorm/ReadVariableOp_12batch_normalization_105/batchnorm/ReadVariableOp_12h
2batch_normalization_105/batchnorm/ReadVariableOp_22batch_normalization_105/batchnorm/ReadVariableOp_22l
4batch_normalization_105/batchnorm/mul/ReadVariableOp4batch_normalization_105/batchnorm/mul/ReadVariableOp2d
0batch_normalization_106/batchnorm/ReadVariableOp0batch_normalization_106/batchnorm/ReadVariableOp2h
2batch_normalization_106/batchnorm/ReadVariableOp_12batch_normalization_106/batchnorm/ReadVariableOp_12h
2batch_normalization_106/batchnorm/ReadVariableOp_22batch_normalization_106/batchnorm/ReadVariableOp_22l
4batch_normalization_106/batchnorm/mul/ReadVariableOp4batch_normalization_106/batchnorm/mul/ReadVariableOp2@
dense_85/MatMul/ReadVariableOpdense_85/MatMul/ReadVariableOp2@
dense_86/MatMul/ReadVariableOpdense_86/MatMul/ReadVariableOp2@
dense_87/MatMul/ReadVariableOpdense_87/MatMul/ReadVariableOp2@
dense_88/MatMul/ReadVariableOpdense_88/MatMul/ReadVariableOp2B
dense_89/BiasAdd/ReadVariableOpdense_89/BiasAdd/ReadVariableOp2@
dense_89/MatMul/ReadVariableOpdense_89/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????

!
_user_specified_name	input_1
?
?
9__inference_feed_forward_sub_net_17_layer_call_fn_7071754
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
T__inference_feed_forward_sub_net_17_layer_call_and_return_conditional_losses_70712152
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
?,
?
T__inference_batch_normalization_103_layer_call_and_return_conditional_losses_7070294

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
%__inference_signature_wrapper_7071526
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
"__inference__wrapped_model_70700422
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
?
?
9__inference_feed_forward_sub_net_17_layer_call_fn_7071640
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
T__inference_feed_forward_sub_net_17_layer_call_and_return_conditional_losses_70709892
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
T__inference_batch_normalization_102_layer_call_and_return_conditional_losses_7072420

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
T__inference_batch_normalization_103_layer_call_and_return_conditional_losses_7072502

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
?"
"__inference__wrapped_model_7070042
input_1_
Qfeed_forward_sub_net_17_batch_normalization_102_batchnorm_readvariableop_resource:
c
Ufeed_forward_sub_net_17_batch_normalization_102_batchnorm_mul_readvariableop_resource:
a
Sfeed_forward_sub_net_17_batch_normalization_102_batchnorm_readvariableop_1_resource:
a
Sfeed_forward_sub_net_17_batch_normalization_102_batchnorm_readvariableop_2_resource:
Q
?feed_forward_sub_net_17_dense_85_matmul_readvariableop_resource:
_
Qfeed_forward_sub_net_17_batch_normalization_103_batchnorm_readvariableop_resource:c
Ufeed_forward_sub_net_17_batch_normalization_103_batchnorm_mul_readvariableop_resource:a
Sfeed_forward_sub_net_17_batch_normalization_103_batchnorm_readvariableop_1_resource:a
Sfeed_forward_sub_net_17_batch_normalization_103_batchnorm_readvariableop_2_resource:Q
?feed_forward_sub_net_17_dense_86_matmul_readvariableop_resource:_
Qfeed_forward_sub_net_17_batch_normalization_104_batchnorm_readvariableop_resource:c
Ufeed_forward_sub_net_17_batch_normalization_104_batchnorm_mul_readvariableop_resource:a
Sfeed_forward_sub_net_17_batch_normalization_104_batchnorm_readvariableop_1_resource:a
Sfeed_forward_sub_net_17_batch_normalization_104_batchnorm_readvariableop_2_resource:Q
?feed_forward_sub_net_17_dense_87_matmul_readvariableop_resource:_
Qfeed_forward_sub_net_17_batch_normalization_105_batchnorm_readvariableop_resource:c
Ufeed_forward_sub_net_17_batch_normalization_105_batchnorm_mul_readvariableop_resource:a
Sfeed_forward_sub_net_17_batch_normalization_105_batchnorm_readvariableop_1_resource:a
Sfeed_forward_sub_net_17_batch_normalization_105_batchnorm_readvariableop_2_resource:Q
?feed_forward_sub_net_17_dense_88_matmul_readvariableop_resource:_
Qfeed_forward_sub_net_17_batch_normalization_106_batchnorm_readvariableop_resource:c
Ufeed_forward_sub_net_17_batch_normalization_106_batchnorm_mul_readvariableop_resource:a
Sfeed_forward_sub_net_17_batch_normalization_106_batchnorm_readvariableop_1_resource:a
Sfeed_forward_sub_net_17_batch_normalization_106_batchnorm_readvariableop_2_resource:Q
?feed_forward_sub_net_17_dense_89_matmul_readvariableop_resource:
N
@feed_forward_sub_net_17_dense_89_biasadd_readvariableop_resource:

identity??Hfeed_forward_sub_net_17/batch_normalization_102/batchnorm/ReadVariableOp?Jfeed_forward_sub_net_17/batch_normalization_102/batchnorm/ReadVariableOp_1?Jfeed_forward_sub_net_17/batch_normalization_102/batchnorm/ReadVariableOp_2?Lfeed_forward_sub_net_17/batch_normalization_102/batchnorm/mul/ReadVariableOp?Hfeed_forward_sub_net_17/batch_normalization_103/batchnorm/ReadVariableOp?Jfeed_forward_sub_net_17/batch_normalization_103/batchnorm/ReadVariableOp_1?Jfeed_forward_sub_net_17/batch_normalization_103/batchnorm/ReadVariableOp_2?Lfeed_forward_sub_net_17/batch_normalization_103/batchnorm/mul/ReadVariableOp?Hfeed_forward_sub_net_17/batch_normalization_104/batchnorm/ReadVariableOp?Jfeed_forward_sub_net_17/batch_normalization_104/batchnorm/ReadVariableOp_1?Jfeed_forward_sub_net_17/batch_normalization_104/batchnorm/ReadVariableOp_2?Lfeed_forward_sub_net_17/batch_normalization_104/batchnorm/mul/ReadVariableOp?Hfeed_forward_sub_net_17/batch_normalization_105/batchnorm/ReadVariableOp?Jfeed_forward_sub_net_17/batch_normalization_105/batchnorm/ReadVariableOp_1?Jfeed_forward_sub_net_17/batch_normalization_105/batchnorm/ReadVariableOp_2?Lfeed_forward_sub_net_17/batch_normalization_105/batchnorm/mul/ReadVariableOp?Hfeed_forward_sub_net_17/batch_normalization_106/batchnorm/ReadVariableOp?Jfeed_forward_sub_net_17/batch_normalization_106/batchnorm/ReadVariableOp_1?Jfeed_forward_sub_net_17/batch_normalization_106/batchnorm/ReadVariableOp_2?Lfeed_forward_sub_net_17/batch_normalization_106/batchnorm/mul/ReadVariableOp?6feed_forward_sub_net_17/dense_85/MatMul/ReadVariableOp?6feed_forward_sub_net_17/dense_86/MatMul/ReadVariableOp?6feed_forward_sub_net_17/dense_87/MatMul/ReadVariableOp?6feed_forward_sub_net_17/dense_88/MatMul/ReadVariableOp?7feed_forward_sub_net_17/dense_89/BiasAdd/ReadVariableOp?6feed_forward_sub_net_17/dense_89/MatMul/ReadVariableOp?
Hfeed_forward_sub_net_17/batch_normalization_102/batchnorm/ReadVariableOpReadVariableOpQfeed_forward_sub_net_17_batch_normalization_102_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype02J
Hfeed_forward_sub_net_17/batch_normalization_102/batchnorm/ReadVariableOp?
?feed_forward_sub_net_17/batch_normalization_102/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2A
?feed_forward_sub_net_17/batch_normalization_102/batchnorm/add/y?
=feed_forward_sub_net_17/batch_normalization_102/batchnorm/addAddV2Pfeed_forward_sub_net_17/batch_normalization_102/batchnorm/ReadVariableOp:value:0Hfeed_forward_sub_net_17/batch_normalization_102/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2?
=feed_forward_sub_net_17/batch_normalization_102/batchnorm/add?
?feed_forward_sub_net_17/batch_normalization_102/batchnorm/RsqrtRsqrtAfeed_forward_sub_net_17/batch_normalization_102/batchnorm/add:z:0*
T0*
_output_shapes
:
2A
?feed_forward_sub_net_17/batch_normalization_102/batchnorm/Rsqrt?
Lfeed_forward_sub_net_17/batch_normalization_102/batchnorm/mul/ReadVariableOpReadVariableOpUfeed_forward_sub_net_17_batch_normalization_102_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype02N
Lfeed_forward_sub_net_17/batch_normalization_102/batchnorm/mul/ReadVariableOp?
=feed_forward_sub_net_17/batch_normalization_102/batchnorm/mulMulCfeed_forward_sub_net_17/batch_normalization_102/batchnorm/Rsqrt:y:0Tfeed_forward_sub_net_17/batch_normalization_102/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2?
=feed_forward_sub_net_17/batch_normalization_102/batchnorm/mul?
?feed_forward_sub_net_17/batch_normalization_102/batchnorm/mul_1Mulinput_1Afeed_forward_sub_net_17/batch_normalization_102/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????
2A
?feed_forward_sub_net_17/batch_normalization_102/batchnorm/mul_1?
Jfeed_forward_sub_net_17/batch_normalization_102/batchnorm/ReadVariableOp_1ReadVariableOpSfeed_forward_sub_net_17_batch_normalization_102_batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype02L
Jfeed_forward_sub_net_17/batch_normalization_102/batchnorm/ReadVariableOp_1?
?feed_forward_sub_net_17/batch_normalization_102/batchnorm/mul_2MulRfeed_forward_sub_net_17/batch_normalization_102/batchnorm/ReadVariableOp_1:value:0Afeed_forward_sub_net_17/batch_normalization_102/batchnorm/mul:z:0*
T0*
_output_shapes
:
2A
?feed_forward_sub_net_17/batch_normalization_102/batchnorm/mul_2?
Jfeed_forward_sub_net_17/batch_normalization_102/batchnorm/ReadVariableOp_2ReadVariableOpSfeed_forward_sub_net_17_batch_normalization_102_batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype02L
Jfeed_forward_sub_net_17/batch_normalization_102/batchnorm/ReadVariableOp_2?
=feed_forward_sub_net_17/batch_normalization_102/batchnorm/subSubRfeed_forward_sub_net_17/batch_normalization_102/batchnorm/ReadVariableOp_2:value:0Cfeed_forward_sub_net_17/batch_normalization_102/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2?
=feed_forward_sub_net_17/batch_normalization_102/batchnorm/sub?
?feed_forward_sub_net_17/batch_normalization_102/batchnorm/add_1AddV2Cfeed_forward_sub_net_17/batch_normalization_102/batchnorm/mul_1:z:0Afeed_forward_sub_net_17/batch_normalization_102/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????
2A
?feed_forward_sub_net_17/batch_normalization_102/batchnorm/add_1?
6feed_forward_sub_net_17/dense_85/MatMul/ReadVariableOpReadVariableOp?feed_forward_sub_net_17_dense_85_matmul_readvariableop_resource*
_output_shapes

:
*
dtype028
6feed_forward_sub_net_17/dense_85/MatMul/ReadVariableOp?
'feed_forward_sub_net_17/dense_85/MatMulMatMulCfeed_forward_sub_net_17/batch_normalization_102/batchnorm/add_1:z:0>feed_forward_sub_net_17/dense_85/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2)
'feed_forward_sub_net_17/dense_85/MatMul?
Hfeed_forward_sub_net_17/batch_normalization_103/batchnorm/ReadVariableOpReadVariableOpQfeed_forward_sub_net_17_batch_normalization_103_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02J
Hfeed_forward_sub_net_17/batch_normalization_103/batchnorm/ReadVariableOp?
?feed_forward_sub_net_17/batch_normalization_103/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2A
?feed_forward_sub_net_17/batch_normalization_103/batchnorm/add/y?
=feed_forward_sub_net_17/batch_normalization_103/batchnorm/addAddV2Pfeed_forward_sub_net_17/batch_normalization_103/batchnorm/ReadVariableOp:value:0Hfeed_forward_sub_net_17/batch_normalization_103/batchnorm/add/y:output:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_17/batch_normalization_103/batchnorm/add?
?feed_forward_sub_net_17/batch_normalization_103/batchnorm/RsqrtRsqrtAfeed_forward_sub_net_17/batch_normalization_103/batchnorm/add:z:0*
T0*
_output_shapes
:2A
?feed_forward_sub_net_17/batch_normalization_103/batchnorm/Rsqrt?
Lfeed_forward_sub_net_17/batch_normalization_103/batchnorm/mul/ReadVariableOpReadVariableOpUfeed_forward_sub_net_17_batch_normalization_103_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02N
Lfeed_forward_sub_net_17/batch_normalization_103/batchnorm/mul/ReadVariableOp?
=feed_forward_sub_net_17/batch_normalization_103/batchnorm/mulMulCfeed_forward_sub_net_17/batch_normalization_103/batchnorm/Rsqrt:y:0Tfeed_forward_sub_net_17/batch_normalization_103/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_17/batch_normalization_103/batchnorm/mul?
?feed_forward_sub_net_17/batch_normalization_103/batchnorm/mul_1Mul1feed_forward_sub_net_17/dense_85/MatMul:product:0Afeed_forward_sub_net_17/batch_normalization_103/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2A
?feed_forward_sub_net_17/batch_normalization_103/batchnorm/mul_1?
Jfeed_forward_sub_net_17/batch_normalization_103/batchnorm/ReadVariableOp_1ReadVariableOpSfeed_forward_sub_net_17_batch_normalization_103_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02L
Jfeed_forward_sub_net_17/batch_normalization_103/batchnorm/ReadVariableOp_1?
?feed_forward_sub_net_17/batch_normalization_103/batchnorm/mul_2MulRfeed_forward_sub_net_17/batch_normalization_103/batchnorm/ReadVariableOp_1:value:0Afeed_forward_sub_net_17/batch_normalization_103/batchnorm/mul:z:0*
T0*
_output_shapes
:2A
?feed_forward_sub_net_17/batch_normalization_103/batchnorm/mul_2?
Jfeed_forward_sub_net_17/batch_normalization_103/batchnorm/ReadVariableOp_2ReadVariableOpSfeed_forward_sub_net_17_batch_normalization_103_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02L
Jfeed_forward_sub_net_17/batch_normalization_103/batchnorm/ReadVariableOp_2?
=feed_forward_sub_net_17/batch_normalization_103/batchnorm/subSubRfeed_forward_sub_net_17/batch_normalization_103/batchnorm/ReadVariableOp_2:value:0Cfeed_forward_sub_net_17/batch_normalization_103/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_17/batch_normalization_103/batchnorm/sub?
?feed_forward_sub_net_17/batch_normalization_103/batchnorm/add_1AddV2Cfeed_forward_sub_net_17/batch_normalization_103/batchnorm/mul_1:z:0Afeed_forward_sub_net_17/batch_normalization_103/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2A
?feed_forward_sub_net_17/batch_normalization_103/batchnorm/add_1?
feed_forward_sub_net_17/ReluReluCfeed_forward_sub_net_17/batch_normalization_103/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
feed_forward_sub_net_17/Relu?
6feed_forward_sub_net_17/dense_86/MatMul/ReadVariableOpReadVariableOp?feed_forward_sub_net_17_dense_86_matmul_readvariableop_resource*
_output_shapes

:*
dtype028
6feed_forward_sub_net_17/dense_86/MatMul/ReadVariableOp?
'feed_forward_sub_net_17/dense_86/MatMulMatMul*feed_forward_sub_net_17/Relu:activations:0>feed_forward_sub_net_17/dense_86/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2)
'feed_forward_sub_net_17/dense_86/MatMul?
Hfeed_forward_sub_net_17/batch_normalization_104/batchnorm/ReadVariableOpReadVariableOpQfeed_forward_sub_net_17_batch_normalization_104_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02J
Hfeed_forward_sub_net_17/batch_normalization_104/batchnorm/ReadVariableOp?
?feed_forward_sub_net_17/batch_normalization_104/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2A
?feed_forward_sub_net_17/batch_normalization_104/batchnorm/add/y?
=feed_forward_sub_net_17/batch_normalization_104/batchnorm/addAddV2Pfeed_forward_sub_net_17/batch_normalization_104/batchnorm/ReadVariableOp:value:0Hfeed_forward_sub_net_17/batch_normalization_104/batchnorm/add/y:output:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_17/batch_normalization_104/batchnorm/add?
?feed_forward_sub_net_17/batch_normalization_104/batchnorm/RsqrtRsqrtAfeed_forward_sub_net_17/batch_normalization_104/batchnorm/add:z:0*
T0*
_output_shapes
:2A
?feed_forward_sub_net_17/batch_normalization_104/batchnorm/Rsqrt?
Lfeed_forward_sub_net_17/batch_normalization_104/batchnorm/mul/ReadVariableOpReadVariableOpUfeed_forward_sub_net_17_batch_normalization_104_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02N
Lfeed_forward_sub_net_17/batch_normalization_104/batchnorm/mul/ReadVariableOp?
=feed_forward_sub_net_17/batch_normalization_104/batchnorm/mulMulCfeed_forward_sub_net_17/batch_normalization_104/batchnorm/Rsqrt:y:0Tfeed_forward_sub_net_17/batch_normalization_104/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_17/batch_normalization_104/batchnorm/mul?
?feed_forward_sub_net_17/batch_normalization_104/batchnorm/mul_1Mul1feed_forward_sub_net_17/dense_86/MatMul:product:0Afeed_forward_sub_net_17/batch_normalization_104/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2A
?feed_forward_sub_net_17/batch_normalization_104/batchnorm/mul_1?
Jfeed_forward_sub_net_17/batch_normalization_104/batchnorm/ReadVariableOp_1ReadVariableOpSfeed_forward_sub_net_17_batch_normalization_104_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02L
Jfeed_forward_sub_net_17/batch_normalization_104/batchnorm/ReadVariableOp_1?
?feed_forward_sub_net_17/batch_normalization_104/batchnorm/mul_2MulRfeed_forward_sub_net_17/batch_normalization_104/batchnorm/ReadVariableOp_1:value:0Afeed_forward_sub_net_17/batch_normalization_104/batchnorm/mul:z:0*
T0*
_output_shapes
:2A
?feed_forward_sub_net_17/batch_normalization_104/batchnorm/mul_2?
Jfeed_forward_sub_net_17/batch_normalization_104/batchnorm/ReadVariableOp_2ReadVariableOpSfeed_forward_sub_net_17_batch_normalization_104_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02L
Jfeed_forward_sub_net_17/batch_normalization_104/batchnorm/ReadVariableOp_2?
=feed_forward_sub_net_17/batch_normalization_104/batchnorm/subSubRfeed_forward_sub_net_17/batch_normalization_104/batchnorm/ReadVariableOp_2:value:0Cfeed_forward_sub_net_17/batch_normalization_104/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_17/batch_normalization_104/batchnorm/sub?
?feed_forward_sub_net_17/batch_normalization_104/batchnorm/add_1AddV2Cfeed_forward_sub_net_17/batch_normalization_104/batchnorm/mul_1:z:0Afeed_forward_sub_net_17/batch_normalization_104/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2A
?feed_forward_sub_net_17/batch_normalization_104/batchnorm/add_1?
feed_forward_sub_net_17/Relu_1ReluCfeed_forward_sub_net_17/batch_normalization_104/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2 
feed_forward_sub_net_17/Relu_1?
6feed_forward_sub_net_17/dense_87/MatMul/ReadVariableOpReadVariableOp?feed_forward_sub_net_17_dense_87_matmul_readvariableop_resource*
_output_shapes

:*
dtype028
6feed_forward_sub_net_17/dense_87/MatMul/ReadVariableOp?
'feed_forward_sub_net_17/dense_87/MatMulMatMul,feed_forward_sub_net_17/Relu_1:activations:0>feed_forward_sub_net_17/dense_87/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2)
'feed_forward_sub_net_17/dense_87/MatMul?
Hfeed_forward_sub_net_17/batch_normalization_105/batchnorm/ReadVariableOpReadVariableOpQfeed_forward_sub_net_17_batch_normalization_105_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02J
Hfeed_forward_sub_net_17/batch_normalization_105/batchnorm/ReadVariableOp?
?feed_forward_sub_net_17/batch_normalization_105/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2A
?feed_forward_sub_net_17/batch_normalization_105/batchnorm/add/y?
=feed_forward_sub_net_17/batch_normalization_105/batchnorm/addAddV2Pfeed_forward_sub_net_17/batch_normalization_105/batchnorm/ReadVariableOp:value:0Hfeed_forward_sub_net_17/batch_normalization_105/batchnorm/add/y:output:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_17/batch_normalization_105/batchnorm/add?
?feed_forward_sub_net_17/batch_normalization_105/batchnorm/RsqrtRsqrtAfeed_forward_sub_net_17/batch_normalization_105/batchnorm/add:z:0*
T0*
_output_shapes
:2A
?feed_forward_sub_net_17/batch_normalization_105/batchnorm/Rsqrt?
Lfeed_forward_sub_net_17/batch_normalization_105/batchnorm/mul/ReadVariableOpReadVariableOpUfeed_forward_sub_net_17_batch_normalization_105_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02N
Lfeed_forward_sub_net_17/batch_normalization_105/batchnorm/mul/ReadVariableOp?
=feed_forward_sub_net_17/batch_normalization_105/batchnorm/mulMulCfeed_forward_sub_net_17/batch_normalization_105/batchnorm/Rsqrt:y:0Tfeed_forward_sub_net_17/batch_normalization_105/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_17/batch_normalization_105/batchnorm/mul?
?feed_forward_sub_net_17/batch_normalization_105/batchnorm/mul_1Mul1feed_forward_sub_net_17/dense_87/MatMul:product:0Afeed_forward_sub_net_17/batch_normalization_105/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2A
?feed_forward_sub_net_17/batch_normalization_105/batchnorm/mul_1?
Jfeed_forward_sub_net_17/batch_normalization_105/batchnorm/ReadVariableOp_1ReadVariableOpSfeed_forward_sub_net_17_batch_normalization_105_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02L
Jfeed_forward_sub_net_17/batch_normalization_105/batchnorm/ReadVariableOp_1?
?feed_forward_sub_net_17/batch_normalization_105/batchnorm/mul_2MulRfeed_forward_sub_net_17/batch_normalization_105/batchnorm/ReadVariableOp_1:value:0Afeed_forward_sub_net_17/batch_normalization_105/batchnorm/mul:z:0*
T0*
_output_shapes
:2A
?feed_forward_sub_net_17/batch_normalization_105/batchnorm/mul_2?
Jfeed_forward_sub_net_17/batch_normalization_105/batchnorm/ReadVariableOp_2ReadVariableOpSfeed_forward_sub_net_17_batch_normalization_105_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02L
Jfeed_forward_sub_net_17/batch_normalization_105/batchnorm/ReadVariableOp_2?
=feed_forward_sub_net_17/batch_normalization_105/batchnorm/subSubRfeed_forward_sub_net_17/batch_normalization_105/batchnorm/ReadVariableOp_2:value:0Cfeed_forward_sub_net_17/batch_normalization_105/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_17/batch_normalization_105/batchnorm/sub?
?feed_forward_sub_net_17/batch_normalization_105/batchnorm/add_1AddV2Cfeed_forward_sub_net_17/batch_normalization_105/batchnorm/mul_1:z:0Afeed_forward_sub_net_17/batch_normalization_105/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2A
?feed_forward_sub_net_17/batch_normalization_105/batchnorm/add_1?
feed_forward_sub_net_17/Relu_2ReluCfeed_forward_sub_net_17/batch_normalization_105/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2 
feed_forward_sub_net_17/Relu_2?
6feed_forward_sub_net_17/dense_88/MatMul/ReadVariableOpReadVariableOp?feed_forward_sub_net_17_dense_88_matmul_readvariableop_resource*
_output_shapes

:*
dtype028
6feed_forward_sub_net_17/dense_88/MatMul/ReadVariableOp?
'feed_forward_sub_net_17/dense_88/MatMulMatMul,feed_forward_sub_net_17/Relu_2:activations:0>feed_forward_sub_net_17/dense_88/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2)
'feed_forward_sub_net_17/dense_88/MatMul?
Hfeed_forward_sub_net_17/batch_normalization_106/batchnorm/ReadVariableOpReadVariableOpQfeed_forward_sub_net_17_batch_normalization_106_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02J
Hfeed_forward_sub_net_17/batch_normalization_106/batchnorm/ReadVariableOp?
?feed_forward_sub_net_17/batch_normalization_106/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2A
?feed_forward_sub_net_17/batch_normalization_106/batchnorm/add/y?
=feed_forward_sub_net_17/batch_normalization_106/batchnorm/addAddV2Pfeed_forward_sub_net_17/batch_normalization_106/batchnorm/ReadVariableOp:value:0Hfeed_forward_sub_net_17/batch_normalization_106/batchnorm/add/y:output:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_17/batch_normalization_106/batchnorm/add?
?feed_forward_sub_net_17/batch_normalization_106/batchnorm/RsqrtRsqrtAfeed_forward_sub_net_17/batch_normalization_106/batchnorm/add:z:0*
T0*
_output_shapes
:2A
?feed_forward_sub_net_17/batch_normalization_106/batchnorm/Rsqrt?
Lfeed_forward_sub_net_17/batch_normalization_106/batchnorm/mul/ReadVariableOpReadVariableOpUfeed_forward_sub_net_17_batch_normalization_106_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02N
Lfeed_forward_sub_net_17/batch_normalization_106/batchnorm/mul/ReadVariableOp?
=feed_forward_sub_net_17/batch_normalization_106/batchnorm/mulMulCfeed_forward_sub_net_17/batch_normalization_106/batchnorm/Rsqrt:y:0Tfeed_forward_sub_net_17/batch_normalization_106/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_17/batch_normalization_106/batchnorm/mul?
?feed_forward_sub_net_17/batch_normalization_106/batchnorm/mul_1Mul1feed_forward_sub_net_17/dense_88/MatMul:product:0Afeed_forward_sub_net_17/batch_normalization_106/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2A
?feed_forward_sub_net_17/batch_normalization_106/batchnorm/mul_1?
Jfeed_forward_sub_net_17/batch_normalization_106/batchnorm/ReadVariableOp_1ReadVariableOpSfeed_forward_sub_net_17_batch_normalization_106_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02L
Jfeed_forward_sub_net_17/batch_normalization_106/batchnorm/ReadVariableOp_1?
?feed_forward_sub_net_17/batch_normalization_106/batchnorm/mul_2MulRfeed_forward_sub_net_17/batch_normalization_106/batchnorm/ReadVariableOp_1:value:0Afeed_forward_sub_net_17/batch_normalization_106/batchnorm/mul:z:0*
T0*
_output_shapes
:2A
?feed_forward_sub_net_17/batch_normalization_106/batchnorm/mul_2?
Jfeed_forward_sub_net_17/batch_normalization_106/batchnorm/ReadVariableOp_2ReadVariableOpSfeed_forward_sub_net_17_batch_normalization_106_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02L
Jfeed_forward_sub_net_17/batch_normalization_106/batchnorm/ReadVariableOp_2?
=feed_forward_sub_net_17/batch_normalization_106/batchnorm/subSubRfeed_forward_sub_net_17/batch_normalization_106/batchnorm/ReadVariableOp_2:value:0Cfeed_forward_sub_net_17/batch_normalization_106/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_17/batch_normalization_106/batchnorm/sub?
?feed_forward_sub_net_17/batch_normalization_106/batchnorm/add_1AddV2Cfeed_forward_sub_net_17/batch_normalization_106/batchnorm/mul_1:z:0Afeed_forward_sub_net_17/batch_normalization_106/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2A
?feed_forward_sub_net_17/batch_normalization_106/batchnorm/add_1?
feed_forward_sub_net_17/Relu_3ReluCfeed_forward_sub_net_17/batch_normalization_106/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2 
feed_forward_sub_net_17/Relu_3?
6feed_forward_sub_net_17/dense_89/MatMul/ReadVariableOpReadVariableOp?feed_forward_sub_net_17_dense_89_matmul_readvariableop_resource*
_output_shapes

:
*
dtype028
6feed_forward_sub_net_17/dense_89/MatMul/ReadVariableOp?
'feed_forward_sub_net_17/dense_89/MatMulMatMul,feed_forward_sub_net_17/Relu_3:activations:0>feed_forward_sub_net_17/dense_89/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2)
'feed_forward_sub_net_17/dense_89/MatMul?
7feed_forward_sub_net_17/dense_89/BiasAdd/ReadVariableOpReadVariableOp@feed_forward_sub_net_17_dense_89_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype029
7feed_forward_sub_net_17/dense_89/BiasAdd/ReadVariableOp?
(feed_forward_sub_net_17/dense_89/BiasAddBiasAdd1feed_forward_sub_net_17/dense_89/MatMul:product:0?feed_forward_sub_net_17/dense_89/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2*
(feed_forward_sub_net_17/dense_89/BiasAdd?
IdentityIdentity1feed_forward_sub_net_17/dense_89/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOpI^feed_forward_sub_net_17/batch_normalization_102/batchnorm/ReadVariableOpK^feed_forward_sub_net_17/batch_normalization_102/batchnorm/ReadVariableOp_1K^feed_forward_sub_net_17/batch_normalization_102/batchnorm/ReadVariableOp_2M^feed_forward_sub_net_17/batch_normalization_102/batchnorm/mul/ReadVariableOpI^feed_forward_sub_net_17/batch_normalization_103/batchnorm/ReadVariableOpK^feed_forward_sub_net_17/batch_normalization_103/batchnorm/ReadVariableOp_1K^feed_forward_sub_net_17/batch_normalization_103/batchnorm/ReadVariableOp_2M^feed_forward_sub_net_17/batch_normalization_103/batchnorm/mul/ReadVariableOpI^feed_forward_sub_net_17/batch_normalization_104/batchnorm/ReadVariableOpK^feed_forward_sub_net_17/batch_normalization_104/batchnorm/ReadVariableOp_1K^feed_forward_sub_net_17/batch_normalization_104/batchnorm/ReadVariableOp_2M^feed_forward_sub_net_17/batch_normalization_104/batchnorm/mul/ReadVariableOpI^feed_forward_sub_net_17/batch_normalization_105/batchnorm/ReadVariableOpK^feed_forward_sub_net_17/batch_normalization_105/batchnorm/ReadVariableOp_1K^feed_forward_sub_net_17/batch_normalization_105/batchnorm/ReadVariableOp_2M^feed_forward_sub_net_17/batch_normalization_105/batchnorm/mul/ReadVariableOpI^feed_forward_sub_net_17/batch_normalization_106/batchnorm/ReadVariableOpK^feed_forward_sub_net_17/batch_normalization_106/batchnorm/ReadVariableOp_1K^feed_forward_sub_net_17/batch_normalization_106/batchnorm/ReadVariableOp_2M^feed_forward_sub_net_17/batch_normalization_106/batchnorm/mul/ReadVariableOp7^feed_forward_sub_net_17/dense_85/MatMul/ReadVariableOp7^feed_forward_sub_net_17/dense_86/MatMul/ReadVariableOp7^feed_forward_sub_net_17/dense_87/MatMul/ReadVariableOp7^feed_forward_sub_net_17/dense_88/MatMul/ReadVariableOp8^feed_forward_sub_net_17/dense_89/BiasAdd/ReadVariableOp7^feed_forward_sub_net_17/dense_89/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????
: : : : : : : : : : : : : : : : : : : : : : : : : : 2?
Hfeed_forward_sub_net_17/batch_normalization_102/batchnorm/ReadVariableOpHfeed_forward_sub_net_17/batch_normalization_102/batchnorm/ReadVariableOp2?
Jfeed_forward_sub_net_17/batch_normalization_102/batchnorm/ReadVariableOp_1Jfeed_forward_sub_net_17/batch_normalization_102/batchnorm/ReadVariableOp_12?
Jfeed_forward_sub_net_17/batch_normalization_102/batchnorm/ReadVariableOp_2Jfeed_forward_sub_net_17/batch_normalization_102/batchnorm/ReadVariableOp_22?
Lfeed_forward_sub_net_17/batch_normalization_102/batchnorm/mul/ReadVariableOpLfeed_forward_sub_net_17/batch_normalization_102/batchnorm/mul/ReadVariableOp2?
Hfeed_forward_sub_net_17/batch_normalization_103/batchnorm/ReadVariableOpHfeed_forward_sub_net_17/batch_normalization_103/batchnorm/ReadVariableOp2?
Jfeed_forward_sub_net_17/batch_normalization_103/batchnorm/ReadVariableOp_1Jfeed_forward_sub_net_17/batch_normalization_103/batchnorm/ReadVariableOp_12?
Jfeed_forward_sub_net_17/batch_normalization_103/batchnorm/ReadVariableOp_2Jfeed_forward_sub_net_17/batch_normalization_103/batchnorm/ReadVariableOp_22?
Lfeed_forward_sub_net_17/batch_normalization_103/batchnorm/mul/ReadVariableOpLfeed_forward_sub_net_17/batch_normalization_103/batchnorm/mul/ReadVariableOp2?
Hfeed_forward_sub_net_17/batch_normalization_104/batchnorm/ReadVariableOpHfeed_forward_sub_net_17/batch_normalization_104/batchnorm/ReadVariableOp2?
Jfeed_forward_sub_net_17/batch_normalization_104/batchnorm/ReadVariableOp_1Jfeed_forward_sub_net_17/batch_normalization_104/batchnorm/ReadVariableOp_12?
Jfeed_forward_sub_net_17/batch_normalization_104/batchnorm/ReadVariableOp_2Jfeed_forward_sub_net_17/batch_normalization_104/batchnorm/ReadVariableOp_22?
Lfeed_forward_sub_net_17/batch_normalization_104/batchnorm/mul/ReadVariableOpLfeed_forward_sub_net_17/batch_normalization_104/batchnorm/mul/ReadVariableOp2?
Hfeed_forward_sub_net_17/batch_normalization_105/batchnorm/ReadVariableOpHfeed_forward_sub_net_17/batch_normalization_105/batchnorm/ReadVariableOp2?
Jfeed_forward_sub_net_17/batch_normalization_105/batchnorm/ReadVariableOp_1Jfeed_forward_sub_net_17/batch_normalization_105/batchnorm/ReadVariableOp_12?
Jfeed_forward_sub_net_17/batch_normalization_105/batchnorm/ReadVariableOp_2Jfeed_forward_sub_net_17/batch_normalization_105/batchnorm/ReadVariableOp_22?
Lfeed_forward_sub_net_17/batch_normalization_105/batchnorm/mul/ReadVariableOpLfeed_forward_sub_net_17/batch_normalization_105/batchnorm/mul/ReadVariableOp2?
Hfeed_forward_sub_net_17/batch_normalization_106/batchnorm/ReadVariableOpHfeed_forward_sub_net_17/batch_normalization_106/batchnorm/ReadVariableOp2?
Jfeed_forward_sub_net_17/batch_normalization_106/batchnorm/ReadVariableOp_1Jfeed_forward_sub_net_17/batch_normalization_106/batchnorm/ReadVariableOp_12?
Jfeed_forward_sub_net_17/batch_normalization_106/batchnorm/ReadVariableOp_2Jfeed_forward_sub_net_17/batch_normalization_106/batchnorm/ReadVariableOp_22?
Lfeed_forward_sub_net_17/batch_normalization_106/batchnorm/mul/ReadVariableOpLfeed_forward_sub_net_17/batch_normalization_106/batchnorm/mul/ReadVariableOp2p
6feed_forward_sub_net_17/dense_85/MatMul/ReadVariableOp6feed_forward_sub_net_17/dense_85/MatMul/ReadVariableOp2p
6feed_forward_sub_net_17/dense_86/MatMul/ReadVariableOp6feed_forward_sub_net_17/dense_86/MatMul/ReadVariableOp2p
6feed_forward_sub_net_17/dense_87/MatMul/ReadVariableOp6feed_forward_sub_net_17/dense_87/MatMul/ReadVariableOp2p
6feed_forward_sub_net_17/dense_88/MatMul/ReadVariableOp6feed_forward_sub_net_17/dense_88/MatMul/ReadVariableOp2r
7feed_forward_sub_net_17/dense_89/BiasAdd/ReadVariableOp7feed_forward_sub_net_17/dense_89/BiasAdd/ReadVariableOp2p
6feed_forward_sub_net_17/dense_89/MatMul/ReadVariableOp6feed_forward_sub_net_17/dense_89/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????

!
_user_specified_name	input_1
?
?
E__inference_dense_85_layer_call_and_return_conditional_losses_7072762

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
9__inference_batch_normalization_102_layer_call_fn_7072351

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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_102_layer_call_and_return_conditional_losses_70700662
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
?E
?
T__inference_feed_forward_sub_net_17_layer_call_and_return_conditional_losses_7070989
x-
batch_normalization_102_7070879:
-
batch_normalization_102_7070881:
-
batch_normalization_102_7070883:
-
batch_normalization_102_7070885:
"
dense_85_7070896:
-
batch_normalization_103_7070899:-
batch_normalization_103_7070901:-
batch_normalization_103_7070903:-
batch_normalization_103_7070905:"
dense_86_7070917:-
batch_normalization_104_7070920:-
batch_normalization_104_7070922:-
batch_normalization_104_7070924:-
batch_normalization_104_7070926:"
dense_87_7070938:-
batch_normalization_105_7070941:-
batch_normalization_105_7070943:-
batch_normalization_105_7070945:-
batch_normalization_105_7070947:"
dense_88_7070959:-
batch_normalization_106_7070962:-
batch_normalization_106_7070964:-
batch_normalization_106_7070966:-
batch_normalization_106_7070968:"
dense_89_7070983:

dense_89_7070985:

identity??/batch_normalization_102/StatefulPartitionedCall?/batch_normalization_103/StatefulPartitionedCall?/batch_normalization_104/StatefulPartitionedCall?/batch_normalization_105/StatefulPartitionedCall?/batch_normalization_106/StatefulPartitionedCall? dense_85/StatefulPartitionedCall? dense_86/StatefulPartitionedCall? dense_87/StatefulPartitionedCall? dense_88/StatefulPartitionedCall? dense_89/StatefulPartitionedCall?
/batch_normalization_102/StatefulPartitionedCallStatefulPartitionedCallxbatch_normalization_102_7070879batch_normalization_102_7070881batch_normalization_102_7070883batch_normalization_102_7070885*
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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_102_layer_call_and_return_conditional_losses_707006621
/batch_normalization_102/StatefulPartitionedCall?
 dense_85/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_102/StatefulPartitionedCall:output:0dense_85_7070896*
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
E__inference_dense_85_layer_call_and_return_conditional_losses_70708952"
 dense_85/StatefulPartitionedCall?
/batch_normalization_103/StatefulPartitionedCallStatefulPartitionedCall)dense_85/StatefulPartitionedCall:output:0batch_normalization_103_7070899batch_normalization_103_7070901batch_normalization_103_7070903batch_normalization_103_7070905*
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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_103_layer_call_and_return_conditional_losses_707023221
/batch_normalization_103/StatefulPartitionedCall?
ReluRelu8batch_normalization_103/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
Relu?
 dense_86/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0dense_86_7070917*
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
E__inference_dense_86_layer_call_and_return_conditional_losses_70709162"
 dense_86/StatefulPartitionedCall?
/batch_normalization_104/StatefulPartitionedCallStatefulPartitionedCall)dense_86/StatefulPartitionedCall:output:0batch_normalization_104_7070920batch_normalization_104_7070922batch_normalization_104_7070924batch_normalization_104_7070926*
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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_104_layer_call_and_return_conditional_losses_707039821
/batch_normalization_104/StatefulPartitionedCall?
Relu_1Relu8batch_normalization_104/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
Relu_1?
 dense_87/StatefulPartitionedCallStatefulPartitionedCallRelu_1:activations:0dense_87_7070938*
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
E__inference_dense_87_layer_call_and_return_conditional_losses_70709372"
 dense_87/StatefulPartitionedCall?
/batch_normalization_105/StatefulPartitionedCallStatefulPartitionedCall)dense_87/StatefulPartitionedCall:output:0batch_normalization_105_7070941batch_normalization_105_7070943batch_normalization_105_7070945batch_normalization_105_7070947*
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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_105_layer_call_and_return_conditional_losses_707056421
/batch_normalization_105/StatefulPartitionedCall?
Relu_2Relu8batch_normalization_105/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
Relu_2?
 dense_88/StatefulPartitionedCallStatefulPartitionedCallRelu_2:activations:0dense_88_7070959*
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
E__inference_dense_88_layer_call_and_return_conditional_losses_70709582"
 dense_88/StatefulPartitionedCall?
/batch_normalization_106/StatefulPartitionedCallStatefulPartitionedCall)dense_88/StatefulPartitionedCall:output:0batch_normalization_106_7070962batch_normalization_106_7070964batch_normalization_106_7070966batch_normalization_106_7070968*
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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_106_layer_call_and_return_conditional_losses_707073021
/batch_normalization_106/StatefulPartitionedCall?
Relu_3Relu8batch_normalization_106/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
Relu_3?
 dense_89/StatefulPartitionedCallStatefulPartitionedCallRelu_3:activations:0dense_89_7070983dense_89_7070985*
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
E__inference_dense_89_layer_call_and_return_conditional_losses_70709822"
 dense_89/StatefulPartitionedCall?
IdentityIdentity)dense_89/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp0^batch_normalization_102/StatefulPartitionedCall0^batch_normalization_103/StatefulPartitionedCall0^batch_normalization_104/StatefulPartitionedCall0^batch_normalization_105/StatefulPartitionedCall0^batch_normalization_106/StatefulPartitionedCall!^dense_85/StatefulPartitionedCall!^dense_86/StatefulPartitionedCall!^dense_87/StatefulPartitionedCall!^dense_88/StatefulPartitionedCall!^dense_89/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????
: : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_102/StatefulPartitionedCall/batch_normalization_102/StatefulPartitionedCall2b
/batch_normalization_103/StatefulPartitionedCall/batch_normalization_103/StatefulPartitionedCall2b
/batch_normalization_104/StatefulPartitionedCall/batch_normalization_104/StatefulPartitionedCall2b
/batch_normalization_105/StatefulPartitionedCall/batch_normalization_105/StatefulPartitionedCall2b
/batch_normalization_106/StatefulPartitionedCall/batch_normalization_106/StatefulPartitionedCall2D
 dense_85/StatefulPartitionedCall dense_85/StatefulPartitionedCall2D
 dense_86/StatefulPartitionedCall dense_86/StatefulPartitionedCall2D
 dense_87/StatefulPartitionedCall dense_87/StatefulPartitionedCall2D
 dense_88/StatefulPartitionedCall dense_88/StatefulPartitionedCall2D
 dense_89/StatefulPartitionedCall dense_89/StatefulPartitionedCall:J F
'
_output_shapes
:?????????


_user_specified_namex
?
?
T__inference_batch_normalization_102_layer_call_and_return_conditional_losses_7072384

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
?,
?
T__inference_batch_normalization_105_layer_call_and_return_conditional_losses_7072666

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
?

?
E__inference_dense_89_layer_call_and_return_conditional_losses_7070982

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
?
?
9__inference_batch_normalization_103_layer_call_fn_7072433

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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_103_layer_call_and_return_conditional_losses_70702322
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
*__inference_dense_87_layer_call_fn_7072783

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
E__inference_dense_87_layer_call_and_return_conditional_losses_70709372
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
E__inference_dense_87_layer_call_and_return_conditional_losses_7072790

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
?
?
E__inference_dense_87_layer_call_and_return_conditional_losses_7070937

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
?
?
E__inference_dense_86_layer_call_and_return_conditional_losses_7072776

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
?
?
E__inference_dense_85_layer_call_and_return_conditional_losses_7070895

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
T__inference_batch_normalization_104_layer_call_and_return_conditional_losses_7070460

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
T__inference_batch_normalization_106_layer_call_and_return_conditional_losses_7070792

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
T__inference_batch_normalization_105_layer_call_and_return_conditional_losses_7070564

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
9__inference_batch_normalization_103_layer_call_fn_7072446

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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_103_layer_call_and_return_conditional_losses_70702942
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
*__inference_dense_88_layer_call_fn_7072797

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
E__inference_dense_88_layer_call_and_return_conditional_losses_70709582
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
?
?
T__inference_batch_normalization_103_layer_call_and_return_conditional_losses_7070232

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
E__inference_dense_86_layer_call_and_return_conditional_losses_7070916

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
T__inference_batch_normalization_104_layer_call_and_return_conditional_losses_7070398

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
T__inference_batch_normalization_104_layer_call_and_return_conditional_losses_7072584

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
T__inference_batch_normalization_105_layer_call_and_return_conditional_losses_7070626

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
E__inference_dense_88_layer_call_and_return_conditional_losses_7070958

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
9__inference_batch_normalization_104_layer_call_fn_7072515

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
GPU 2J 8? *]
fXRV
T__inference_batch_normalization_104_layer_call_and_return_conditional_losses_70703982
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
T__inference_feed_forward_sub_net_17_layer_call_and_return_conditional_losses_7072338
input_1M
?batch_normalization_102_assignmovingavg_readvariableop_resource:
O
Abatch_normalization_102_assignmovingavg_1_readvariableop_resource:
K
=batch_normalization_102_batchnorm_mul_readvariableop_resource:
G
9batch_normalization_102_batchnorm_readvariableop_resource:
9
'dense_85_matmul_readvariableop_resource:
M
?batch_normalization_103_assignmovingavg_readvariableop_resource:O
Abatch_normalization_103_assignmovingavg_1_readvariableop_resource:K
=batch_normalization_103_batchnorm_mul_readvariableop_resource:G
9batch_normalization_103_batchnorm_readvariableop_resource:9
'dense_86_matmul_readvariableop_resource:M
?batch_normalization_104_assignmovingavg_readvariableop_resource:O
Abatch_normalization_104_assignmovingavg_1_readvariableop_resource:K
=batch_normalization_104_batchnorm_mul_readvariableop_resource:G
9batch_normalization_104_batchnorm_readvariableop_resource:9
'dense_87_matmul_readvariableop_resource:M
?batch_normalization_105_assignmovingavg_readvariableop_resource:O
Abatch_normalization_105_assignmovingavg_1_readvariableop_resource:K
=batch_normalization_105_batchnorm_mul_readvariableop_resource:G
9batch_normalization_105_batchnorm_readvariableop_resource:9
'dense_88_matmul_readvariableop_resource:M
?batch_normalization_106_assignmovingavg_readvariableop_resource:O
Abatch_normalization_106_assignmovingavg_1_readvariableop_resource:K
=batch_normalization_106_batchnorm_mul_readvariableop_resource:G
9batch_normalization_106_batchnorm_readvariableop_resource:9
'dense_89_matmul_readvariableop_resource:
6
(dense_89_biasadd_readvariableop_resource:

identity??'batch_normalization_102/AssignMovingAvg?6batch_normalization_102/AssignMovingAvg/ReadVariableOp?)batch_normalization_102/AssignMovingAvg_1?8batch_normalization_102/AssignMovingAvg_1/ReadVariableOp?0batch_normalization_102/batchnorm/ReadVariableOp?4batch_normalization_102/batchnorm/mul/ReadVariableOp?'batch_normalization_103/AssignMovingAvg?6batch_normalization_103/AssignMovingAvg/ReadVariableOp?)batch_normalization_103/AssignMovingAvg_1?8batch_normalization_103/AssignMovingAvg_1/ReadVariableOp?0batch_normalization_103/batchnorm/ReadVariableOp?4batch_normalization_103/batchnorm/mul/ReadVariableOp?'batch_normalization_104/AssignMovingAvg?6batch_normalization_104/AssignMovingAvg/ReadVariableOp?)batch_normalization_104/AssignMovingAvg_1?8batch_normalization_104/AssignMovingAvg_1/ReadVariableOp?0batch_normalization_104/batchnorm/ReadVariableOp?4batch_normalization_104/batchnorm/mul/ReadVariableOp?'batch_normalization_105/AssignMovingAvg?6batch_normalization_105/AssignMovingAvg/ReadVariableOp?)batch_normalization_105/AssignMovingAvg_1?8batch_normalization_105/AssignMovingAvg_1/ReadVariableOp?0batch_normalization_105/batchnorm/ReadVariableOp?4batch_normalization_105/batchnorm/mul/ReadVariableOp?'batch_normalization_106/AssignMovingAvg?6batch_normalization_106/AssignMovingAvg/ReadVariableOp?)batch_normalization_106/AssignMovingAvg_1?8batch_normalization_106/AssignMovingAvg_1/ReadVariableOp?0batch_normalization_106/batchnorm/ReadVariableOp?4batch_normalization_106/batchnorm/mul/ReadVariableOp?dense_85/MatMul/ReadVariableOp?dense_86/MatMul/ReadVariableOp?dense_87/MatMul/ReadVariableOp?dense_88/MatMul/ReadVariableOp?dense_89/BiasAdd/ReadVariableOp?dense_89/MatMul/ReadVariableOp?
6batch_normalization_102/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 28
6batch_normalization_102/moments/mean/reduction_indices?
$batch_normalization_102/moments/meanMeaninput_1?batch_normalization_102/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2&
$batch_normalization_102/moments/mean?
,batch_normalization_102/moments/StopGradientStopGradient-batch_normalization_102/moments/mean:output:0*
T0*
_output_shapes

:
2.
,batch_normalization_102/moments/StopGradient?
1batch_normalization_102/moments/SquaredDifferenceSquaredDifferenceinput_15batch_normalization_102/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????
23
1batch_normalization_102/moments/SquaredDifference?
:batch_normalization_102/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2<
:batch_normalization_102/moments/variance/reduction_indices?
(batch_normalization_102/moments/varianceMean5batch_normalization_102/moments/SquaredDifference:z:0Cbatch_normalization_102/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2*
(batch_normalization_102/moments/variance?
'batch_normalization_102/moments/SqueezeSqueeze-batch_normalization_102/moments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2)
'batch_normalization_102/moments/Squeeze?
)batch_normalization_102/moments/Squeeze_1Squeeze1batch_normalization_102/moments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2+
)batch_normalization_102/moments/Squeeze_1?
-batch_normalization_102/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-batch_normalization_102/AssignMovingAvg/decay?
,batch_normalization_102/AssignMovingAvg/CastCast6batch_normalization_102/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2.
,batch_normalization_102/AssignMovingAvg/Cast?
6batch_normalization_102/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_102_assignmovingavg_readvariableop_resource*
_output_shapes
:
*
dtype028
6batch_normalization_102/AssignMovingAvg/ReadVariableOp?
+batch_normalization_102/AssignMovingAvg/subSub>batch_normalization_102/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_102/moments/Squeeze:output:0*
T0*
_output_shapes
:
2-
+batch_normalization_102/AssignMovingAvg/sub?
+batch_normalization_102/AssignMovingAvg/mulMul/batch_normalization_102/AssignMovingAvg/sub:z:00batch_normalization_102/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:
2-
+batch_normalization_102/AssignMovingAvg/mul?
'batch_normalization_102/AssignMovingAvgAssignSubVariableOp?batch_normalization_102_assignmovingavg_readvariableop_resource/batch_normalization_102/AssignMovingAvg/mul:z:07^batch_normalization_102/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_102/AssignMovingAvg?
/batch_normalization_102/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<21
/batch_normalization_102/AssignMovingAvg_1/decay?
.batch_normalization_102/AssignMovingAvg_1/CastCast8batch_normalization_102/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 20
.batch_normalization_102/AssignMovingAvg_1/Cast?
8batch_normalization_102/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_102_assignmovingavg_1_readvariableop_resource*
_output_shapes
:
*
dtype02:
8batch_normalization_102/AssignMovingAvg_1/ReadVariableOp?
-batch_normalization_102/AssignMovingAvg_1/subSub@batch_normalization_102/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_102/moments/Squeeze_1:output:0*
T0*
_output_shapes
:
2/
-batch_normalization_102/AssignMovingAvg_1/sub?
-batch_normalization_102/AssignMovingAvg_1/mulMul1batch_normalization_102/AssignMovingAvg_1/sub:z:02batch_normalization_102/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:
2/
-batch_normalization_102/AssignMovingAvg_1/mul?
)batch_normalization_102/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_102_assignmovingavg_1_readvariableop_resource1batch_normalization_102/AssignMovingAvg_1/mul:z:09^batch_normalization_102/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02+
)batch_normalization_102/AssignMovingAvg_1?
'batch_normalization_102/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2)
'batch_normalization_102/batchnorm/add/y?
%batch_normalization_102/batchnorm/addAddV22batch_normalization_102/moments/Squeeze_1:output:00batch_normalization_102/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2'
%batch_normalization_102/batchnorm/add?
'batch_normalization_102/batchnorm/RsqrtRsqrt)batch_normalization_102/batchnorm/add:z:0*
T0*
_output_shapes
:
2)
'batch_normalization_102/batchnorm/Rsqrt?
4batch_normalization_102/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_102_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype026
4batch_normalization_102/batchnorm/mul/ReadVariableOp?
%batch_normalization_102/batchnorm/mulMul+batch_normalization_102/batchnorm/Rsqrt:y:0<batch_normalization_102/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2'
%batch_normalization_102/batchnorm/mul?
'batch_normalization_102/batchnorm/mul_1Mulinput_1)batch_normalization_102/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????
2)
'batch_normalization_102/batchnorm/mul_1?
'batch_normalization_102/batchnorm/mul_2Mul0batch_normalization_102/moments/Squeeze:output:0)batch_normalization_102/batchnorm/mul:z:0*
T0*
_output_shapes
:
2)
'batch_normalization_102/batchnorm/mul_2?
0batch_normalization_102/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_102_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype022
0batch_normalization_102/batchnorm/ReadVariableOp?
%batch_normalization_102/batchnorm/subSub8batch_normalization_102/batchnorm/ReadVariableOp:value:0+batch_normalization_102/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2'
%batch_normalization_102/batchnorm/sub?
'batch_normalization_102/batchnorm/add_1AddV2+batch_normalization_102/batchnorm/mul_1:z:0)batch_normalization_102/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????
2)
'batch_normalization_102/batchnorm/add_1?
dense_85/MatMul/ReadVariableOpReadVariableOp'dense_85_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_85/MatMul/ReadVariableOp?
dense_85/MatMulMatMul+batch_normalization_102/batchnorm/add_1:z:0&dense_85/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_85/MatMul?
6batch_normalization_103/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 28
6batch_normalization_103/moments/mean/reduction_indices?
$batch_normalization_103/moments/meanMeandense_85/MatMul:product:0?batch_normalization_103/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2&
$batch_normalization_103/moments/mean?
,batch_normalization_103/moments/StopGradientStopGradient-batch_normalization_103/moments/mean:output:0*
T0*
_output_shapes

:2.
,batch_normalization_103/moments/StopGradient?
1batch_normalization_103/moments/SquaredDifferenceSquaredDifferencedense_85/MatMul:product:05batch_normalization_103/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????23
1batch_normalization_103/moments/SquaredDifference?
:batch_normalization_103/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2<
:batch_normalization_103/moments/variance/reduction_indices?
(batch_normalization_103/moments/varianceMean5batch_normalization_103/moments/SquaredDifference:z:0Cbatch_normalization_103/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2*
(batch_normalization_103/moments/variance?
'batch_normalization_103/moments/SqueezeSqueeze-batch_normalization_103/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_103/moments/Squeeze?
)batch_normalization_103/moments/Squeeze_1Squeeze1batch_normalization_103/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2+
)batch_normalization_103/moments/Squeeze_1?
-batch_normalization_103/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-batch_normalization_103/AssignMovingAvg/decay?
,batch_normalization_103/AssignMovingAvg/CastCast6batch_normalization_103/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2.
,batch_normalization_103/AssignMovingAvg/Cast?
6batch_normalization_103/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_103_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_103/AssignMovingAvg/ReadVariableOp?
+batch_normalization_103/AssignMovingAvg/subSub>batch_normalization_103/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_103/moments/Squeeze:output:0*
T0*
_output_shapes
:2-
+batch_normalization_103/AssignMovingAvg/sub?
+batch_normalization_103/AssignMovingAvg/mulMul/batch_normalization_103/AssignMovingAvg/sub:z:00batch_normalization_103/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2-
+batch_normalization_103/AssignMovingAvg/mul?
'batch_normalization_103/AssignMovingAvgAssignSubVariableOp?batch_normalization_103_assignmovingavg_readvariableop_resource/batch_normalization_103/AssignMovingAvg/mul:z:07^batch_normalization_103/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_103/AssignMovingAvg?
/batch_normalization_103/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<21
/batch_normalization_103/AssignMovingAvg_1/decay?
.batch_normalization_103/AssignMovingAvg_1/CastCast8batch_normalization_103/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 20
.batch_normalization_103/AssignMovingAvg_1/Cast?
8batch_normalization_103/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_103_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02:
8batch_normalization_103/AssignMovingAvg_1/ReadVariableOp?
-batch_normalization_103/AssignMovingAvg_1/subSub@batch_normalization_103/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_103/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2/
-batch_normalization_103/AssignMovingAvg_1/sub?
-batch_normalization_103/AssignMovingAvg_1/mulMul1batch_normalization_103/AssignMovingAvg_1/sub:z:02batch_normalization_103/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2/
-batch_normalization_103/AssignMovingAvg_1/mul?
)batch_normalization_103/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_103_assignmovingavg_1_readvariableop_resource1batch_normalization_103/AssignMovingAvg_1/mul:z:09^batch_normalization_103/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02+
)batch_normalization_103/AssignMovingAvg_1?
'batch_normalization_103/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2)
'batch_normalization_103/batchnorm/add/y?
%batch_normalization_103/batchnorm/addAddV22batch_normalization_103/moments/Squeeze_1:output:00batch_normalization_103/batchnorm/add/y:output:0*
T0*
_output_shapes
:2'
%batch_normalization_103/batchnorm/add?
'batch_normalization_103/batchnorm/RsqrtRsqrt)batch_normalization_103/batchnorm/add:z:0*
T0*
_output_shapes
:2)
'batch_normalization_103/batchnorm/Rsqrt?
4batch_normalization_103/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_103_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype026
4batch_normalization_103/batchnorm/mul/ReadVariableOp?
%batch_normalization_103/batchnorm/mulMul+batch_normalization_103/batchnorm/Rsqrt:y:0<batch_normalization_103/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2'
%batch_normalization_103/batchnorm/mul?
'batch_normalization_103/batchnorm/mul_1Muldense_85/MatMul:product:0)batch_normalization_103/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2)
'batch_normalization_103/batchnorm/mul_1?
'batch_normalization_103/batchnorm/mul_2Mul0batch_normalization_103/moments/Squeeze:output:0)batch_normalization_103/batchnorm/mul:z:0*
T0*
_output_shapes
:2)
'batch_normalization_103/batchnorm/mul_2?
0batch_normalization_103/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_103_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype022
0batch_normalization_103/batchnorm/ReadVariableOp?
%batch_normalization_103/batchnorm/subSub8batch_normalization_103/batchnorm/ReadVariableOp:value:0+batch_normalization_103/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2'
%batch_normalization_103/batchnorm/sub?
'batch_normalization_103/batchnorm/add_1AddV2+batch_normalization_103/batchnorm/mul_1:z:0)batch_normalization_103/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2)
'batch_normalization_103/batchnorm/add_1s
ReluRelu+batch_normalization_103/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu?
dense_86/MatMul/ReadVariableOpReadVariableOp'dense_86_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_86/MatMul/ReadVariableOp?
dense_86/MatMulMatMulRelu:activations:0&dense_86/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_86/MatMul?
6batch_normalization_104/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 28
6batch_normalization_104/moments/mean/reduction_indices?
$batch_normalization_104/moments/meanMeandense_86/MatMul:product:0?batch_normalization_104/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2&
$batch_normalization_104/moments/mean?
,batch_normalization_104/moments/StopGradientStopGradient-batch_normalization_104/moments/mean:output:0*
T0*
_output_shapes

:2.
,batch_normalization_104/moments/StopGradient?
1batch_normalization_104/moments/SquaredDifferenceSquaredDifferencedense_86/MatMul:product:05batch_normalization_104/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????23
1batch_normalization_104/moments/SquaredDifference?
:batch_normalization_104/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2<
:batch_normalization_104/moments/variance/reduction_indices?
(batch_normalization_104/moments/varianceMean5batch_normalization_104/moments/SquaredDifference:z:0Cbatch_normalization_104/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2*
(batch_normalization_104/moments/variance?
'batch_normalization_104/moments/SqueezeSqueeze-batch_normalization_104/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_104/moments/Squeeze?
)batch_normalization_104/moments/Squeeze_1Squeeze1batch_normalization_104/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2+
)batch_normalization_104/moments/Squeeze_1?
-batch_normalization_104/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-batch_normalization_104/AssignMovingAvg/decay?
,batch_normalization_104/AssignMovingAvg/CastCast6batch_normalization_104/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2.
,batch_normalization_104/AssignMovingAvg/Cast?
6batch_normalization_104/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_104_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_104/AssignMovingAvg/ReadVariableOp?
+batch_normalization_104/AssignMovingAvg/subSub>batch_normalization_104/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_104/moments/Squeeze:output:0*
T0*
_output_shapes
:2-
+batch_normalization_104/AssignMovingAvg/sub?
+batch_normalization_104/AssignMovingAvg/mulMul/batch_normalization_104/AssignMovingAvg/sub:z:00batch_normalization_104/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2-
+batch_normalization_104/AssignMovingAvg/mul?
'batch_normalization_104/AssignMovingAvgAssignSubVariableOp?batch_normalization_104_assignmovingavg_readvariableop_resource/batch_normalization_104/AssignMovingAvg/mul:z:07^batch_normalization_104/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_104/AssignMovingAvg?
/batch_normalization_104/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<21
/batch_normalization_104/AssignMovingAvg_1/decay?
.batch_normalization_104/AssignMovingAvg_1/CastCast8batch_normalization_104/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 20
.batch_normalization_104/AssignMovingAvg_1/Cast?
8batch_normalization_104/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_104_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02:
8batch_normalization_104/AssignMovingAvg_1/ReadVariableOp?
-batch_normalization_104/AssignMovingAvg_1/subSub@batch_normalization_104/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_104/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2/
-batch_normalization_104/AssignMovingAvg_1/sub?
-batch_normalization_104/AssignMovingAvg_1/mulMul1batch_normalization_104/AssignMovingAvg_1/sub:z:02batch_normalization_104/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2/
-batch_normalization_104/AssignMovingAvg_1/mul?
)batch_normalization_104/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_104_assignmovingavg_1_readvariableop_resource1batch_normalization_104/AssignMovingAvg_1/mul:z:09^batch_normalization_104/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02+
)batch_normalization_104/AssignMovingAvg_1?
'batch_normalization_104/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2)
'batch_normalization_104/batchnorm/add/y?
%batch_normalization_104/batchnorm/addAddV22batch_normalization_104/moments/Squeeze_1:output:00batch_normalization_104/batchnorm/add/y:output:0*
T0*
_output_shapes
:2'
%batch_normalization_104/batchnorm/add?
'batch_normalization_104/batchnorm/RsqrtRsqrt)batch_normalization_104/batchnorm/add:z:0*
T0*
_output_shapes
:2)
'batch_normalization_104/batchnorm/Rsqrt?
4batch_normalization_104/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_104_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype026
4batch_normalization_104/batchnorm/mul/ReadVariableOp?
%batch_normalization_104/batchnorm/mulMul+batch_normalization_104/batchnorm/Rsqrt:y:0<batch_normalization_104/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2'
%batch_normalization_104/batchnorm/mul?
'batch_normalization_104/batchnorm/mul_1Muldense_86/MatMul:product:0)batch_normalization_104/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2)
'batch_normalization_104/batchnorm/mul_1?
'batch_normalization_104/batchnorm/mul_2Mul0batch_normalization_104/moments/Squeeze:output:0)batch_normalization_104/batchnorm/mul:z:0*
T0*
_output_shapes
:2)
'batch_normalization_104/batchnorm/mul_2?
0batch_normalization_104/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_104_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype022
0batch_normalization_104/batchnorm/ReadVariableOp?
%batch_normalization_104/batchnorm/subSub8batch_normalization_104/batchnorm/ReadVariableOp:value:0+batch_normalization_104/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2'
%batch_normalization_104/batchnorm/sub?
'batch_normalization_104/batchnorm/add_1AddV2+batch_normalization_104/batchnorm/mul_1:z:0)batch_normalization_104/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2)
'batch_normalization_104/batchnorm/add_1w
Relu_1Relu+batch_normalization_104/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu_1?
dense_87/MatMul/ReadVariableOpReadVariableOp'dense_87_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_87/MatMul/ReadVariableOp?
dense_87/MatMulMatMulRelu_1:activations:0&dense_87/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_87/MatMul?
6batch_normalization_105/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 28
6batch_normalization_105/moments/mean/reduction_indices?
$batch_normalization_105/moments/meanMeandense_87/MatMul:product:0?batch_normalization_105/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2&
$batch_normalization_105/moments/mean?
,batch_normalization_105/moments/StopGradientStopGradient-batch_normalization_105/moments/mean:output:0*
T0*
_output_shapes

:2.
,batch_normalization_105/moments/StopGradient?
1batch_normalization_105/moments/SquaredDifferenceSquaredDifferencedense_87/MatMul:product:05batch_normalization_105/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????23
1batch_normalization_105/moments/SquaredDifference?
:batch_normalization_105/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2<
:batch_normalization_105/moments/variance/reduction_indices?
(batch_normalization_105/moments/varianceMean5batch_normalization_105/moments/SquaredDifference:z:0Cbatch_normalization_105/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2*
(batch_normalization_105/moments/variance?
'batch_normalization_105/moments/SqueezeSqueeze-batch_normalization_105/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_105/moments/Squeeze?
)batch_normalization_105/moments/Squeeze_1Squeeze1batch_normalization_105/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2+
)batch_normalization_105/moments/Squeeze_1?
-batch_normalization_105/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-batch_normalization_105/AssignMovingAvg/decay?
,batch_normalization_105/AssignMovingAvg/CastCast6batch_normalization_105/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2.
,batch_normalization_105/AssignMovingAvg/Cast?
6batch_normalization_105/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_105_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_105/AssignMovingAvg/ReadVariableOp?
+batch_normalization_105/AssignMovingAvg/subSub>batch_normalization_105/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_105/moments/Squeeze:output:0*
T0*
_output_shapes
:2-
+batch_normalization_105/AssignMovingAvg/sub?
+batch_normalization_105/AssignMovingAvg/mulMul/batch_normalization_105/AssignMovingAvg/sub:z:00batch_normalization_105/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2-
+batch_normalization_105/AssignMovingAvg/mul?
'batch_normalization_105/AssignMovingAvgAssignSubVariableOp?batch_normalization_105_assignmovingavg_readvariableop_resource/batch_normalization_105/AssignMovingAvg/mul:z:07^batch_normalization_105/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_105/AssignMovingAvg?
/batch_normalization_105/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<21
/batch_normalization_105/AssignMovingAvg_1/decay?
.batch_normalization_105/AssignMovingAvg_1/CastCast8batch_normalization_105/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 20
.batch_normalization_105/AssignMovingAvg_1/Cast?
8batch_normalization_105/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_105_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02:
8batch_normalization_105/AssignMovingAvg_1/ReadVariableOp?
-batch_normalization_105/AssignMovingAvg_1/subSub@batch_normalization_105/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_105/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2/
-batch_normalization_105/AssignMovingAvg_1/sub?
-batch_normalization_105/AssignMovingAvg_1/mulMul1batch_normalization_105/AssignMovingAvg_1/sub:z:02batch_normalization_105/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2/
-batch_normalization_105/AssignMovingAvg_1/mul?
)batch_normalization_105/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_105_assignmovingavg_1_readvariableop_resource1batch_normalization_105/AssignMovingAvg_1/mul:z:09^batch_normalization_105/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02+
)batch_normalization_105/AssignMovingAvg_1?
'batch_normalization_105/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2)
'batch_normalization_105/batchnorm/add/y?
%batch_normalization_105/batchnorm/addAddV22batch_normalization_105/moments/Squeeze_1:output:00batch_normalization_105/batchnorm/add/y:output:0*
T0*
_output_shapes
:2'
%batch_normalization_105/batchnorm/add?
'batch_normalization_105/batchnorm/RsqrtRsqrt)batch_normalization_105/batchnorm/add:z:0*
T0*
_output_shapes
:2)
'batch_normalization_105/batchnorm/Rsqrt?
4batch_normalization_105/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_105_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype026
4batch_normalization_105/batchnorm/mul/ReadVariableOp?
%batch_normalization_105/batchnorm/mulMul+batch_normalization_105/batchnorm/Rsqrt:y:0<batch_normalization_105/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2'
%batch_normalization_105/batchnorm/mul?
'batch_normalization_105/batchnorm/mul_1Muldense_87/MatMul:product:0)batch_normalization_105/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2)
'batch_normalization_105/batchnorm/mul_1?
'batch_normalization_105/batchnorm/mul_2Mul0batch_normalization_105/moments/Squeeze:output:0)batch_normalization_105/batchnorm/mul:z:0*
T0*
_output_shapes
:2)
'batch_normalization_105/batchnorm/mul_2?
0batch_normalization_105/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_105_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype022
0batch_normalization_105/batchnorm/ReadVariableOp?
%batch_normalization_105/batchnorm/subSub8batch_normalization_105/batchnorm/ReadVariableOp:value:0+batch_normalization_105/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2'
%batch_normalization_105/batchnorm/sub?
'batch_normalization_105/batchnorm/add_1AddV2+batch_normalization_105/batchnorm/mul_1:z:0)batch_normalization_105/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2)
'batch_normalization_105/batchnorm/add_1w
Relu_2Relu+batch_normalization_105/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu_2?
dense_88/MatMul/ReadVariableOpReadVariableOp'dense_88_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_88/MatMul/ReadVariableOp?
dense_88/MatMulMatMulRelu_2:activations:0&dense_88/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_88/MatMul?
6batch_normalization_106/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 28
6batch_normalization_106/moments/mean/reduction_indices?
$batch_normalization_106/moments/meanMeandense_88/MatMul:product:0?batch_normalization_106/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2&
$batch_normalization_106/moments/mean?
,batch_normalization_106/moments/StopGradientStopGradient-batch_normalization_106/moments/mean:output:0*
T0*
_output_shapes

:2.
,batch_normalization_106/moments/StopGradient?
1batch_normalization_106/moments/SquaredDifferenceSquaredDifferencedense_88/MatMul:product:05batch_normalization_106/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????23
1batch_normalization_106/moments/SquaredDifference?
:batch_normalization_106/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2<
:batch_normalization_106/moments/variance/reduction_indices?
(batch_normalization_106/moments/varianceMean5batch_normalization_106/moments/SquaredDifference:z:0Cbatch_normalization_106/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2*
(batch_normalization_106/moments/variance?
'batch_normalization_106/moments/SqueezeSqueeze-batch_normalization_106/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_106/moments/Squeeze?
)batch_normalization_106/moments/Squeeze_1Squeeze1batch_normalization_106/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2+
)batch_normalization_106/moments/Squeeze_1?
-batch_normalization_106/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-batch_normalization_106/AssignMovingAvg/decay?
,batch_normalization_106/AssignMovingAvg/CastCast6batch_normalization_106/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2.
,batch_normalization_106/AssignMovingAvg/Cast?
6batch_normalization_106/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_106_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_106/AssignMovingAvg/ReadVariableOp?
+batch_normalization_106/AssignMovingAvg/subSub>batch_normalization_106/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_106/moments/Squeeze:output:0*
T0*
_output_shapes
:2-
+batch_normalization_106/AssignMovingAvg/sub?
+batch_normalization_106/AssignMovingAvg/mulMul/batch_normalization_106/AssignMovingAvg/sub:z:00batch_normalization_106/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2-
+batch_normalization_106/AssignMovingAvg/mul?
'batch_normalization_106/AssignMovingAvgAssignSubVariableOp?batch_normalization_106_assignmovingavg_readvariableop_resource/batch_normalization_106/AssignMovingAvg/mul:z:07^batch_normalization_106/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_106/AssignMovingAvg?
/batch_normalization_106/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<21
/batch_normalization_106/AssignMovingAvg_1/decay?
.batch_normalization_106/AssignMovingAvg_1/CastCast8batch_normalization_106/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 20
.batch_normalization_106/AssignMovingAvg_1/Cast?
8batch_normalization_106/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_106_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02:
8batch_normalization_106/AssignMovingAvg_1/ReadVariableOp?
-batch_normalization_106/AssignMovingAvg_1/subSub@batch_normalization_106/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_106/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2/
-batch_normalization_106/AssignMovingAvg_1/sub?
-batch_normalization_106/AssignMovingAvg_1/mulMul1batch_normalization_106/AssignMovingAvg_1/sub:z:02batch_normalization_106/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2/
-batch_normalization_106/AssignMovingAvg_1/mul?
)batch_normalization_106/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_106_assignmovingavg_1_readvariableop_resource1batch_normalization_106/AssignMovingAvg_1/mul:z:09^batch_normalization_106/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02+
)batch_normalization_106/AssignMovingAvg_1?
'batch_normalization_106/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2?????ư>2)
'batch_normalization_106/batchnorm/add/y?
%batch_normalization_106/batchnorm/addAddV22batch_normalization_106/moments/Squeeze_1:output:00batch_normalization_106/batchnorm/add/y:output:0*
T0*
_output_shapes
:2'
%batch_normalization_106/batchnorm/add?
'batch_normalization_106/batchnorm/RsqrtRsqrt)batch_normalization_106/batchnorm/add:z:0*
T0*
_output_shapes
:2)
'batch_normalization_106/batchnorm/Rsqrt?
4batch_normalization_106/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_106_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype026
4batch_normalization_106/batchnorm/mul/ReadVariableOp?
%batch_normalization_106/batchnorm/mulMul+batch_normalization_106/batchnorm/Rsqrt:y:0<batch_normalization_106/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2'
%batch_normalization_106/batchnorm/mul?
'batch_normalization_106/batchnorm/mul_1Muldense_88/MatMul:product:0)batch_normalization_106/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2)
'batch_normalization_106/batchnorm/mul_1?
'batch_normalization_106/batchnorm/mul_2Mul0batch_normalization_106/moments/Squeeze:output:0)batch_normalization_106/batchnorm/mul:z:0*
T0*
_output_shapes
:2)
'batch_normalization_106/batchnorm/mul_2?
0batch_normalization_106/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_106_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype022
0batch_normalization_106/batchnorm/ReadVariableOp?
%batch_normalization_106/batchnorm/subSub8batch_normalization_106/batchnorm/ReadVariableOp:value:0+batch_normalization_106/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2'
%batch_normalization_106/batchnorm/sub?
'batch_normalization_106/batchnorm/add_1AddV2+batch_normalization_106/batchnorm/mul_1:z:0)batch_normalization_106/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2)
'batch_normalization_106/batchnorm/add_1w
Relu_3Relu+batch_normalization_106/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu_3?
dense_89/MatMul/ReadVariableOpReadVariableOp'dense_89_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_89/MatMul/ReadVariableOp?
dense_89/MatMulMatMulRelu_3:activations:0&dense_89/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_89/MatMul?
dense_89/BiasAdd/ReadVariableOpReadVariableOp(dense_89_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_89/BiasAdd/ReadVariableOp?
dense_89/BiasAddBiasAdddense_89/MatMul:product:0'dense_89/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_89/BiasAddt
IdentityIdentitydense_89/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp(^batch_normalization_102/AssignMovingAvg7^batch_normalization_102/AssignMovingAvg/ReadVariableOp*^batch_normalization_102/AssignMovingAvg_19^batch_normalization_102/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_102/batchnorm/ReadVariableOp5^batch_normalization_102/batchnorm/mul/ReadVariableOp(^batch_normalization_103/AssignMovingAvg7^batch_normalization_103/AssignMovingAvg/ReadVariableOp*^batch_normalization_103/AssignMovingAvg_19^batch_normalization_103/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_103/batchnorm/ReadVariableOp5^batch_normalization_103/batchnorm/mul/ReadVariableOp(^batch_normalization_104/AssignMovingAvg7^batch_normalization_104/AssignMovingAvg/ReadVariableOp*^batch_normalization_104/AssignMovingAvg_19^batch_normalization_104/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_104/batchnorm/ReadVariableOp5^batch_normalization_104/batchnorm/mul/ReadVariableOp(^batch_normalization_105/AssignMovingAvg7^batch_normalization_105/AssignMovingAvg/ReadVariableOp*^batch_normalization_105/AssignMovingAvg_19^batch_normalization_105/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_105/batchnorm/ReadVariableOp5^batch_normalization_105/batchnorm/mul/ReadVariableOp(^batch_normalization_106/AssignMovingAvg7^batch_normalization_106/AssignMovingAvg/ReadVariableOp*^batch_normalization_106/AssignMovingAvg_19^batch_normalization_106/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_106/batchnorm/ReadVariableOp5^batch_normalization_106/batchnorm/mul/ReadVariableOp^dense_85/MatMul/ReadVariableOp^dense_86/MatMul/ReadVariableOp^dense_87/MatMul/ReadVariableOp^dense_88/MatMul/ReadVariableOp ^dense_89/BiasAdd/ReadVariableOp^dense_89/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????
: : : : : : : : : : : : : : : : : : : : : : : : : : 2R
'batch_normalization_102/AssignMovingAvg'batch_normalization_102/AssignMovingAvg2p
6batch_normalization_102/AssignMovingAvg/ReadVariableOp6batch_normalization_102/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_102/AssignMovingAvg_1)batch_normalization_102/AssignMovingAvg_12t
8batch_normalization_102/AssignMovingAvg_1/ReadVariableOp8batch_normalization_102/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_102/batchnorm/ReadVariableOp0batch_normalization_102/batchnorm/ReadVariableOp2l
4batch_normalization_102/batchnorm/mul/ReadVariableOp4batch_normalization_102/batchnorm/mul/ReadVariableOp2R
'batch_normalization_103/AssignMovingAvg'batch_normalization_103/AssignMovingAvg2p
6batch_normalization_103/AssignMovingAvg/ReadVariableOp6batch_normalization_103/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_103/AssignMovingAvg_1)batch_normalization_103/AssignMovingAvg_12t
8batch_normalization_103/AssignMovingAvg_1/ReadVariableOp8batch_normalization_103/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_103/batchnorm/ReadVariableOp0batch_normalization_103/batchnorm/ReadVariableOp2l
4batch_normalization_103/batchnorm/mul/ReadVariableOp4batch_normalization_103/batchnorm/mul/ReadVariableOp2R
'batch_normalization_104/AssignMovingAvg'batch_normalization_104/AssignMovingAvg2p
6batch_normalization_104/AssignMovingAvg/ReadVariableOp6batch_normalization_104/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_104/AssignMovingAvg_1)batch_normalization_104/AssignMovingAvg_12t
8batch_normalization_104/AssignMovingAvg_1/ReadVariableOp8batch_normalization_104/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_104/batchnorm/ReadVariableOp0batch_normalization_104/batchnorm/ReadVariableOp2l
4batch_normalization_104/batchnorm/mul/ReadVariableOp4batch_normalization_104/batchnorm/mul/ReadVariableOp2R
'batch_normalization_105/AssignMovingAvg'batch_normalization_105/AssignMovingAvg2p
6batch_normalization_105/AssignMovingAvg/ReadVariableOp6batch_normalization_105/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_105/AssignMovingAvg_1)batch_normalization_105/AssignMovingAvg_12t
8batch_normalization_105/AssignMovingAvg_1/ReadVariableOp8batch_normalization_105/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_105/batchnorm/ReadVariableOp0batch_normalization_105/batchnorm/ReadVariableOp2l
4batch_normalization_105/batchnorm/mul/ReadVariableOp4batch_normalization_105/batchnorm/mul/ReadVariableOp2R
'batch_normalization_106/AssignMovingAvg'batch_normalization_106/AssignMovingAvg2p
6batch_normalization_106/AssignMovingAvg/ReadVariableOp6batch_normalization_106/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_106/AssignMovingAvg_1)batch_normalization_106/AssignMovingAvg_12t
8batch_normalization_106/AssignMovingAvg_1/ReadVariableOp8batch_normalization_106/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_106/batchnorm/ReadVariableOp0batch_normalization_106/batchnorm/ReadVariableOp2l
4batch_normalization_106/batchnorm/mul/ReadVariableOp4batch_normalization_106/batchnorm/mul/ReadVariableOp2@
dense_85/MatMul/ReadVariableOpdense_85/MatMul/ReadVariableOp2@
dense_86/MatMul/ReadVariableOpdense_86/MatMul/ReadVariableOp2@
dense_87/MatMul/ReadVariableOpdense_87/MatMul/ReadVariableOp2@
dense_88/MatMul/ReadVariableOpdense_88/MatMul/ReadVariableOp2B
dense_89/BiasAdd/ReadVariableOpdense_89/BiasAdd/ReadVariableOp2@
dense_89/MatMul/ReadVariableOpdense_89/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????

!
_user_specified_name	input_1
?
?
T__inference_batch_normalization_105_layer_call_and_return_conditional_losses_7072630

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
?
?
9__inference_feed_forward_sub_net_17_layer_call_fn_7071583
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
T__inference_feed_forward_sub_net_17_layer_call_and_return_conditional_losses_70709892
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
U:S
2Gnonshared_model_1/feed_forward_sub_net_17/batch_normalization_102/gamma
T:R
2Fnonshared_model_1/feed_forward_sub_net_17/batch_normalization_102/beta
U:S2Gnonshared_model_1/feed_forward_sub_net_17/batch_normalization_103/gamma
T:R2Fnonshared_model_1/feed_forward_sub_net_17/batch_normalization_103/beta
U:S2Gnonshared_model_1/feed_forward_sub_net_17/batch_normalization_104/gamma
T:R2Fnonshared_model_1/feed_forward_sub_net_17/batch_normalization_104/beta
U:S2Gnonshared_model_1/feed_forward_sub_net_17/batch_normalization_105/gamma
T:R2Fnonshared_model_1/feed_forward_sub_net_17/batch_normalization_105/beta
U:S2Gnonshared_model_1/feed_forward_sub_net_17/batch_normalization_106/gamma
T:R2Fnonshared_model_1/feed_forward_sub_net_17/batch_normalization_106/beta
]:[
 (2Mnonshared_model_1/feed_forward_sub_net_17/batch_normalization_102/moving_mean
a:_
 (2Qnonshared_model_1/feed_forward_sub_net_17/batch_normalization_102/moving_variance
]:[ (2Mnonshared_model_1/feed_forward_sub_net_17/batch_normalization_103/moving_mean
a:_ (2Qnonshared_model_1/feed_forward_sub_net_17/batch_normalization_103/moving_variance
]:[ (2Mnonshared_model_1/feed_forward_sub_net_17/batch_normalization_104/moving_mean
a:_ (2Qnonshared_model_1/feed_forward_sub_net_17/batch_normalization_104/moving_variance
]:[ (2Mnonshared_model_1/feed_forward_sub_net_17/batch_normalization_105/moving_mean
a:_ (2Qnonshared_model_1/feed_forward_sub_net_17/batch_normalization_105/moving_variance
]:[ (2Mnonshared_model_1/feed_forward_sub_net_17/batch_normalization_106/moving_mean
a:_ (2Qnonshared_model_1/feed_forward_sub_net_17/batch_normalization_106/moving_variance
K:I
29nonshared_model_1/feed_forward_sub_net_17/dense_85/kernel
K:I29nonshared_model_1/feed_forward_sub_net_17/dense_86/kernel
K:I29nonshared_model_1/feed_forward_sub_net_17/dense_87/kernel
K:I29nonshared_model_1/feed_forward_sub_net_17/dense_88/kernel
K:I
29nonshared_model_1/feed_forward_sub_net_17/dense_89/kernel
E:C
27nonshared_model_1/feed_forward_sub_net_17/dense_89/bias
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
"__inference__wrapped_model_7070042input_1"?
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
9__inference_feed_forward_sub_net_17_layer_call_fn_7071583
9__inference_feed_forward_sub_net_17_layer_call_fn_7071640
9__inference_feed_forward_sub_net_17_layer_call_fn_7071697
9__inference_feed_forward_sub_net_17_layer_call_fn_7071754?
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
T__inference_feed_forward_sub_net_17_layer_call_and_return_conditional_losses_7071860
T__inference_feed_forward_sub_net_17_layer_call_and_return_conditional_losses_7072046
T__inference_feed_forward_sub_net_17_layer_call_and_return_conditional_losses_7072152
T__inference_feed_forward_sub_net_17_layer_call_and_return_conditional_losses_7072338?
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
%__inference_signature_wrapper_7071526input_1"?
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
9__inference_batch_normalization_102_layer_call_fn_7072351
9__inference_batch_normalization_102_layer_call_fn_7072364?
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
T__inference_batch_normalization_102_layer_call_and_return_conditional_losses_7072384
T__inference_batch_normalization_102_layer_call_and_return_conditional_losses_7072420?
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
9__inference_batch_normalization_103_layer_call_fn_7072433
9__inference_batch_normalization_103_layer_call_fn_7072446?
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
T__inference_batch_normalization_103_layer_call_and_return_conditional_losses_7072466
T__inference_batch_normalization_103_layer_call_and_return_conditional_losses_7072502?
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
9__inference_batch_normalization_104_layer_call_fn_7072515
9__inference_batch_normalization_104_layer_call_fn_7072528?
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
T__inference_batch_normalization_104_layer_call_and_return_conditional_losses_7072548
T__inference_batch_normalization_104_layer_call_and_return_conditional_losses_7072584?
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
9__inference_batch_normalization_105_layer_call_fn_7072597
9__inference_batch_normalization_105_layer_call_fn_7072610?
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
T__inference_batch_normalization_105_layer_call_and_return_conditional_losses_7072630
T__inference_batch_normalization_105_layer_call_and_return_conditional_losses_7072666?
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
9__inference_batch_normalization_106_layer_call_fn_7072679
9__inference_batch_normalization_106_layer_call_fn_7072692?
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
T__inference_batch_normalization_106_layer_call_and_return_conditional_losses_7072712
T__inference_batch_normalization_106_layer_call_and_return_conditional_losses_7072748?
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
*__inference_dense_85_layer_call_fn_7072755?
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
E__inference_dense_85_layer_call_and_return_conditional_losses_7072762?
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
*__inference_dense_86_layer_call_fn_7072769?
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
E__inference_dense_86_layer_call_and_return_conditional_losses_7072776?
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
*__inference_dense_87_layer_call_fn_7072783?
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
E__inference_dense_87_layer_call_and_return_conditional_losses_7072790?
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
*__inference_dense_88_layer_call_fn_7072797?
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
E__inference_dense_88_layer_call_and_return_conditional_losses_7072804?
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
*__inference_dense_89_layer_call_fn_7072813?
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
E__inference_dense_89_layer_call_and_return_conditional_losses_7072823?
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
"__inference__wrapped_model_7070042?' ("!)$#*&%+,0?-
&?#
!?
input_1?????????

? "3?0
.
output_1"?
output_1?????????
?
T__inference_batch_normalization_102_layer_call_and_return_conditional_losses_7072384b3?0
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
T__inference_batch_normalization_102_layer_call_and_return_conditional_losses_7072420b3?0
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
9__inference_batch_normalization_102_layer_call_fn_7072351U3?0
)?&
 ?
inputs?????????

p 
? "??????????
?
9__inference_batch_normalization_102_layer_call_fn_7072364U3?0
)?&
 ?
inputs?????????

p
? "??????????
?
T__inference_batch_normalization_103_layer_call_and_return_conditional_losses_7072466b 3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
T__inference_batch_normalization_103_layer_call_and_return_conditional_losses_7072502b 3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? ?
9__inference_batch_normalization_103_layer_call_fn_7072433U 3?0
)?&
 ?
inputs?????????
p 
? "???????????
9__inference_batch_normalization_103_layer_call_fn_7072446U 3?0
)?&
 ?
inputs?????????
p
? "???????????
T__inference_batch_normalization_104_layer_call_and_return_conditional_losses_7072548b"!3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
T__inference_batch_normalization_104_layer_call_and_return_conditional_losses_7072584b!"3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? ?
9__inference_batch_normalization_104_layer_call_fn_7072515U"!3?0
)?&
 ?
inputs?????????
p 
? "???????????
9__inference_batch_normalization_104_layer_call_fn_7072528U!"3?0
)?&
 ?
inputs?????????
p
? "???????????
T__inference_batch_normalization_105_layer_call_and_return_conditional_losses_7072630b$#3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
T__inference_batch_normalization_105_layer_call_and_return_conditional_losses_7072666b#$3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? ?
9__inference_batch_normalization_105_layer_call_fn_7072597U$#3?0
)?&
 ?
inputs?????????
p 
? "???????????
9__inference_batch_normalization_105_layer_call_fn_7072610U#$3?0
)?&
 ?
inputs?????????
p
? "???????????
T__inference_batch_normalization_106_layer_call_and_return_conditional_losses_7072712b&%3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
T__inference_batch_normalization_106_layer_call_and_return_conditional_losses_7072748b%&3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? ?
9__inference_batch_normalization_106_layer_call_fn_7072679U&%3?0
)?&
 ?
inputs?????????
p 
? "???????????
9__inference_batch_normalization_106_layer_call_fn_7072692U%&3?0
)?&
 ?
inputs?????????
p
? "???????????
E__inference_dense_85_layer_call_and_return_conditional_losses_7072762['/?,
%?"
 ?
inputs?????????

? "%?"
?
0?????????
? |
*__inference_dense_85_layer_call_fn_7072755N'/?,
%?"
 ?
inputs?????????

? "???????????
E__inference_dense_86_layer_call_and_return_conditional_losses_7072776[(/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
*__inference_dense_86_layer_call_fn_7072769N(/?,
%?"
 ?
inputs?????????
? "???????????
E__inference_dense_87_layer_call_and_return_conditional_losses_7072790[)/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
*__inference_dense_87_layer_call_fn_7072783N)/?,
%?"
 ?
inputs?????????
? "???????????
E__inference_dense_88_layer_call_and_return_conditional_losses_7072804[*/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
*__inference_dense_88_layer_call_fn_7072797N*/?,
%?"
 ?
inputs?????????
? "???????????
E__inference_dense_89_layer_call_and_return_conditional_losses_7072823\+,/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????

? }
*__inference_dense_89_layer_call_fn_7072813O+,/?,
%?"
 ?
inputs?????????
? "??????????
?
T__inference_feed_forward_sub_net_17_layer_call_and_return_conditional_losses_7071860s' ("!)$#*&%+,.?+
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
T__inference_feed_forward_sub_net_17_layer_call_and_return_conditional_losses_7072046s' (!")#$*%&+,.?+
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
T__inference_feed_forward_sub_net_17_layer_call_and_return_conditional_losses_7072152y' ("!)$#*&%+,4?1
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
T__inference_feed_forward_sub_net_17_layer_call_and_return_conditional_losses_7072338y' (!")#$*%&+,4?1
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
9__inference_feed_forward_sub_net_17_layer_call_fn_7071583l' ("!)$#*&%+,4?1
*?'
!?
input_1?????????

p 
? "??????????
?
9__inference_feed_forward_sub_net_17_layer_call_fn_7071640f' ("!)$#*&%+,.?+
$?!
?
x?????????

p 
? "??????????
?
9__inference_feed_forward_sub_net_17_layer_call_fn_7071697f' (!")#$*%&+,.?+
$?!
?
x?????????

p
? "??????????
?
9__inference_feed_forward_sub_net_17_layer_call_fn_7071754l' (!")#$*%&+,4?1
*?'
!?
input_1?????????

p
? "??????????
?
%__inference_signature_wrapper_7071526?' ("!)$#*&%+,;?8
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