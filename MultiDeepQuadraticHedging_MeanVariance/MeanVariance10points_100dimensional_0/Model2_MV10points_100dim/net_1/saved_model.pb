ð
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
 "serve*2.6.02unknown8³
å
Fnonshared_model_1/feed_forward_sub_net_10/batch_normalization_60/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:È*W
shared_nameHFnonshared_model_1/feed_forward_sub_net_10/batch_normalization_60/gamma
Þ
Znonshared_model_1/feed_forward_sub_net_10/batch_normalization_60/gamma/Read/ReadVariableOpReadVariableOpFnonshared_model_1/feed_forward_sub_net_10/batch_normalization_60/gamma*
_output_shapes	
:È*
dtype0
ã
Enonshared_model_1/feed_forward_sub_net_10/batch_normalization_60/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:È*V
shared_nameGEnonshared_model_1/feed_forward_sub_net_10/batch_normalization_60/beta
Ü
Ynonshared_model_1/feed_forward_sub_net_10/batch_normalization_60/beta/Read/ReadVariableOpReadVariableOpEnonshared_model_1/feed_forward_sub_net_10/batch_normalization_60/beta*
_output_shapes	
:È*
dtype0
å
Fnonshared_model_1/feed_forward_sub_net_10/batch_normalization_61/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ü*W
shared_nameHFnonshared_model_1/feed_forward_sub_net_10/batch_normalization_61/gamma
Þ
Znonshared_model_1/feed_forward_sub_net_10/batch_normalization_61/gamma/Read/ReadVariableOpReadVariableOpFnonshared_model_1/feed_forward_sub_net_10/batch_normalization_61/gamma*
_output_shapes	
:Ü*
dtype0
ã
Enonshared_model_1/feed_forward_sub_net_10/batch_normalization_61/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ü*V
shared_nameGEnonshared_model_1/feed_forward_sub_net_10/batch_normalization_61/beta
Ü
Ynonshared_model_1/feed_forward_sub_net_10/batch_normalization_61/beta/Read/ReadVariableOpReadVariableOpEnonshared_model_1/feed_forward_sub_net_10/batch_normalization_61/beta*
_output_shapes	
:Ü*
dtype0
å
Fnonshared_model_1/feed_forward_sub_net_10/batch_normalization_62/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ü*W
shared_nameHFnonshared_model_1/feed_forward_sub_net_10/batch_normalization_62/gamma
Þ
Znonshared_model_1/feed_forward_sub_net_10/batch_normalization_62/gamma/Read/ReadVariableOpReadVariableOpFnonshared_model_1/feed_forward_sub_net_10/batch_normalization_62/gamma*
_output_shapes	
:Ü*
dtype0
ã
Enonshared_model_1/feed_forward_sub_net_10/batch_normalization_62/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ü*V
shared_nameGEnonshared_model_1/feed_forward_sub_net_10/batch_normalization_62/beta
Ü
Ynonshared_model_1/feed_forward_sub_net_10/batch_normalization_62/beta/Read/ReadVariableOpReadVariableOpEnonshared_model_1/feed_forward_sub_net_10/batch_normalization_62/beta*
_output_shapes	
:Ü*
dtype0
å
Fnonshared_model_1/feed_forward_sub_net_10/batch_normalization_63/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ü*W
shared_nameHFnonshared_model_1/feed_forward_sub_net_10/batch_normalization_63/gamma
Þ
Znonshared_model_1/feed_forward_sub_net_10/batch_normalization_63/gamma/Read/ReadVariableOpReadVariableOpFnonshared_model_1/feed_forward_sub_net_10/batch_normalization_63/gamma*
_output_shapes	
:Ü*
dtype0
ã
Enonshared_model_1/feed_forward_sub_net_10/batch_normalization_63/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ü*V
shared_nameGEnonshared_model_1/feed_forward_sub_net_10/batch_normalization_63/beta
Ü
Ynonshared_model_1/feed_forward_sub_net_10/batch_normalization_63/beta/Read/ReadVariableOpReadVariableOpEnonshared_model_1/feed_forward_sub_net_10/batch_normalization_63/beta*
_output_shapes	
:Ü*
dtype0
å
Fnonshared_model_1/feed_forward_sub_net_10/batch_normalization_64/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ü*W
shared_nameHFnonshared_model_1/feed_forward_sub_net_10/batch_normalization_64/gamma
Þ
Znonshared_model_1/feed_forward_sub_net_10/batch_normalization_64/gamma/Read/ReadVariableOpReadVariableOpFnonshared_model_1/feed_forward_sub_net_10/batch_normalization_64/gamma*
_output_shapes	
:Ü*
dtype0
ã
Enonshared_model_1/feed_forward_sub_net_10/batch_normalization_64/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ü*V
shared_nameGEnonshared_model_1/feed_forward_sub_net_10/batch_normalization_64/beta
Ü
Ynonshared_model_1/feed_forward_sub_net_10/batch_normalization_64/beta/Read/ReadVariableOpReadVariableOpEnonshared_model_1/feed_forward_sub_net_10/batch_normalization_64/beta*
_output_shapes	
:Ü*
dtype0
Ð
9nonshared_model_1/feed_forward_sub_net_10/dense_50/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ÈÜ*J
shared_name;9nonshared_model_1/feed_forward_sub_net_10/dense_50/kernel
É
Mnonshared_model_1/feed_forward_sub_net_10/dense_50/kernel/Read/ReadVariableOpReadVariableOp9nonshared_model_1/feed_forward_sub_net_10/dense_50/kernel* 
_output_shapes
:
ÈÜ*
dtype0
Ð
9nonshared_model_1/feed_forward_sub_net_10/dense_51/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ÜÜ*J
shared_name;9nonshared_model_1/feed_forward_sub_net_10/dense_51/kernel
É
Mnonshared_model_1/feed_forward_sub_net_10/dense_51/kernel/Read/ReadVariableOpReadVariableOp9nonshared_model_1/feed_forward_sub_net_10/dense_51/kernel* 
_output_shapes
:
ÜÜ*
dtype0
Ð
9nonshared_model_1/feed_forward_sub_net_10/dense_52/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ÜÜ*J
shared_name;9nonshared_model_1/feed_forward_sub_net_10/dense_52/kernel
É
Mnonshared_model_1/feed_forward_sub_net_10/dense_52/kernel/Read/ReadVariableOpReadVariableOp9nonshared_model_1/feed_forward_sub_net_10/dense_52/kernel* 
_output_shapes
:
ÜÜ*
dtype0
Ð
9nonshared_model_1/feed_forward_sub_net_10/dense_53/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ÜÜ*J
shared_name;9nonshared_model_1/feed_forward_sub_net_10/dense_53/kernel
É
Mnonshared_model_1/feed_forward_sub_net_10/dense_53/kernel/Read/ReadVariableOpReadVariableOp9nonshared_model_1/feed_forward_sub_net_10/dense_53/kernel* 
_output_shapes
:
ÜÜ*
dtype0
Ð
9nonshared_model_1/feed_forward_sub_net_10/dense_54/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ÜÈ*J
shared_name;9nonshared_model_1/feed_forward_sub_net_10/dense_54/kernel
É
Mnonshared_model_1/feed_forward_sub_net_10/dense_54/kernel/Read/ReadVariableOpReadVariableOp9nonshared_model_1/feed_forward_sub_net_10/dense_54/kernel* 
_output_shapes
:
ÜÈ*
dtype0
Ç
7nonshared_model_1/feed_forward_sub_net_10/dense_54/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:È*H
shared_name97nonshared_model_1/feed_forward_sub_net_10/dense_54/bias
À
Knonshared_model_1/feed_forward_sub_net_10/dense_54/bias/Read/ReadVariableOpReadVariableOp7nonshared_model_1/feed_forward_sub_net_10/dense_54/bias*
_output_shapes	
:È*
dtype0
ñ
Lnonshared_model_1/feed_forward_sub_net_10/batch_normalization_60/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:È*]
shared_nameNLnonshared_model_1/feed_forward_sub_net_10/batch_normalization_60/moving_mean
ê
`nonshared_model_1/feed_forward_sub_net_10/batch_normalization_60/moving_mean/Read/ReadVariableOpReadVariableOpLnonshared_model_1/feed_forward_sub_net_10/batch_normalization_60/moving_mean*
_output_shapes	
:È*
dtype0
ù
Pnonshared_model_1/feed_forward_sub_net_10/batch_normalization_60/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:È*a
shared_nameRPnonshared_model_1/feed_forward_sub_net_10/batch_normalization_60/moving_variance
ò
dnonshared_model_1/feed_forward_sub_net_10/batch_normalization_60/moving_variance/Read/ReadVariableOpReadVariableOpPnonshared_model_1/feed_forward_sub_net_10/batch_normalization_60/moving_variance*
_output_shapes	
:È*
dtype0
ñ
Lnonshared_model_1/feed_forward_sub_net_10/batch_normalization_61/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ü*]
shared_nameNLnonshared_model_1/feed_forward_sub_net_10/batch_normalization_61/moving_mean
ê
`nonshared_model_1/feed_forward_sub_net_10/batch_normalization_61/moving_mean/Read/ReadVariableOpReadVariableOpLnonshared_model_1/feed_forward_sub_net_10/batch_normalization_61/moving_mean*
_output_shapes	
:Ü*
dtype0
ù
Pnonshared_model_1/feed_forward_sub_net_10/batch_normalization_61/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ü*a
shared_nameRPnonshared_model_1/feed_forward_sub_net_10/batch_normalization_61/moving_variance
ò
dnonshared_model_1/feed_forward_sub_net_10/batch_normalization_61/moving_variance/Read/ReadVariableOpReadVariableOpPnonshared_model_1/feed_forward_sub_net_10/batch_normalization_61/moving_variance*
_output_shapes	
:Ü*
dtype0
ñ
Lnonshared_model_1/feed_forward_sub_net_10/batch_normalization_62/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ü*]
shared_nameNLnonshared_model_1/feed_forward_sub_net_10/batch_normalization_62/moving_mean
ê
`nonshared_model_1/feed_forward_sub_net_10/batch_normalization_62/moving_mean/Read/ReadVariableOpReadVariableOpLnonshared_model_1/feed_forward_sub_net_10/batch_normalization_62/moving_mean*
_output_shapes	
:Ü*
dtype0
ù
Pnonshared_model_1/feed_forward_sub_net_10/batch_normalization_62/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ü*a
shared_nameRPnonshared_model_1/feed_forward_sub_net_10/batch_normalization_62/moving_variance
ò
dnonshared_model_1/feed_forward_sub_net_10/batch_normalization_62/moving_variance/Read/ReadVariableOpReadVariableOpPnonshared_model_1/feed_forward_sub_net_10/batch_normalization_62/moving_variance*
_output_shapes	
:Ü*
dtype0
ñ
Lnonshared_model_1/feed_forward_sub_net_10/batch_normalization_63/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ü*]
shared_nameNLnonshared_model_1/feed_forward_sub_net_10/batch_normalization_63/moving_mean
ê
`nonshared_model_1/feed_forward_sub_net_10/batch_normalization_63/moving_mean/Read/ReadVariableOpReadVariableOpLnonshared_model_1/feed_forward_sub_net_10/batch_normalization_63/moving_mean*
_output_shapes	
:Ü*
dtype0
ù
Pnonshared_model_1/feed_forward_sub_net_10/batch_normalization_63/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ü*a
shared_nameRPnonshared_model_1/feed_forward_sub_net_10/batch_normalization_63/moving_variance
ò
dnonshared_model_1/feed_forward_sub_net_10/batch_normalization_63/moving_variance/Read/ReadVariableOpReadVariableOpPnonshared_model_1/feed_forward_sub_net_10/batch_normalization_63/moving_variance*
_output_shapes	
:Ü*
dtype0
ñ
Lnonshared_model_1/feed_forward_sub_net_10/batch_normalization_64/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ü*]
shared_nameNLnonshared_model_1/feed_forward_sub_net_10/batch_normalization_64/moving_mean
ê
`nonshared_model_1/feed_forward_sub_net_10/batch_normalization_64/moving_mean/Read/ReadVariableOpReadVariableOpLnonshared_model_1/feed_forward_sub_net_10/batch_normalization_64/moving_mean*
_output_shapes	
:Ü*
dtype0
ù
Pnonshared_model_1/feed_forward_sub_net_10/batch_normalization_64/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ü*a
shared_nameRPnonshared_model_1/feed_forward_sub_net_10/batch_normalization_64/moving_variance
ò
dnonshared_model_1/feed_forward_sub_net_10/batch_normalization_64/moving_variance/Read/ReadVariableOpReadVariableOpPnonshared_model_1/feed_forward_sub_net_10/batch_normalization_64/moving_variance*
_output_shapes	
:Ü*
dtype0

NoOpNoOp
@
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Å?
value»?B¸? B±?

	bn_layers
dense_layers
trainable_variables
regularization_losses
	variables
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
10
11
12
 13
!14
"15
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
#10
$11
%12
&13
'14
(15
)16
*17
+18
,19
20
21
22
 23
!24
"25
­
trainable_variables

-layers
regularization_losses
.metrics
	variables
/non_trainable_variables
0layer_regularization_losses
1layer_metrics
 

2axis
	gamma
beta
#moving_mean
$moving_variance
3trainable_variables
4regularization_losses
5	variables
6	keras_api

7axis
	gamma
beta
%moving_mean
&moving_variance
8trainable_variables
9regularization_losses
:	variables
;	keras_api

<axis
	gamma
beta
'moving_mean
(moving_variance
=trainable_variables
>regularization_losses
?	variables
@	keras_api

Aaxis
	gamma
beta
)moving_mean
*moving_variance
Btrainable_variables
Cregularization_losses
D	variables
E	keras_api

Faxis
	gamma
beta
+moving_mean
,moving_variance
Gtrainable_variables
Hregularization_losses
I	variables
J	keras_api

K	keras_api
^

kernel
Ltrainable_variables
Mregularization_losses
N	variables
O	keras_api
^

kernel
Ptrainable_variables
Qregularization_losses
R	variables
S	keras_api
^

kernel
Ttrainable_variables
Uregularization_losses
V	variables
W	keras_api
^

 kernel
Xtrainable_variables
Yregularization_losses
Z	variables
[	keras_api
h

!kernel
"bias
\trainable_variables
]regularization_losses
^	variables
_	keras_api

VARIABLE_VALUEFnonshared_model_1/feed_forward_sub_net_10/batch_normalization_60/gamma0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEEnonshared_model_1/feed_forward_sub_net_10/batch_normalization_60/beta0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEFnonshared_model_1/feed_forward_sub_net_10/batch_normalization_61/gamma0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEEnonshared_model_1/feed_forward_sub_net_10/batch_normalization_61/beta0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEFnonshared_model_1/feed_forward_sub_net_10/batch_normalization_62/gamma0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEEnonshared_model_1/feed_forward_sub_net_10/batch_normalization_62/beta0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEFnonshared_model_1/feed_forward_sub_net_10/batch_normalization_63/gamma0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEEnonshared_model_1/feed_forward_sub_net_10/batch_normalization_63/beta0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEFnonshared_model_1/feed_forward_sub_net_10/batch_normalization_64/gamma0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEEnonshared_model_1/feed_forward_sub_net_10/batch_normalization_64/beta0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUE9nonshared_model_1/feed_forward_sub_net_10/dense_50/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUE9nonshared_model_1/feed_forward_sub_net_10/dense_51/kernel1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUE9nonshared_model_1/feed_forward_sub_net_10/dense_52/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUE9nonshared_model_1/feed_forward_sub_net_10/dense_53/kernel1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUE9nonshared_model_1/feed_forward_sub_net_10/dense_54/kernel1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE7nonshared_model_1/feed_forward_sub_net_10/dense_54/bias1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUELnonshared_model_1/feed_forward_sub_net_10/batch_normalization_60/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEPnonshared_model_1/feed_forward_sub_net_10/batch_normalization_60/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUELnonshared_model_1/feed_forward_sub_net_10/batch_normalization_61/moving_mean'variables/12/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEPnonshared_model_1/feed_forward_sub_net_10/batch_normalization_61/moving_variance'variables/13/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUELnonshared_model_1/feed_forward_sub_net_10/batch_normalization_62/moving_mean'variables/14/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEPnonshared_model_1/feed_forward_sub_net_10/batch_normalization_62/moving_variance'variables/15/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUELnonshared_model_1/feed_forward_sub_net_10/batch_normalization_63/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEPnonshared_model_1/feed_forward_sub_net_10/batch_normalization_63/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUELnonshared_model_1/feed_forward_sub_net_10/batch_normalization_64/moving_mean'variables/18/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEPnonshared_model_1/feed_forward_sub_net_10/batch_normalization_64/moving_variance'variables/19/.ATTRIBUTES/VARIABLE_VALUE
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
F
#0
$1
%2
&3
'4
(5
)6
*7
+8
,9
 
 
 

0
1
 

0
1
#2
$3
­
3trainable_variables

`layers
4regularization_losses
ametrics
5	variables
bnon_trainable_variables
clayer_regularization_losses
dlayer_metrics
 

0
1
 

0
1
%2
&3
­
8trainable_variables

elayers
9regularization_losses
fmetrics
:	variables
gnon_trainable_variables
hlayer_regularization_losses
ilayer_metrics
 

0
1
 

0
1
'2
(3
­
=trainable_variables

jlayers
>regularization_losses
kmetrics
?	variables
lnon_trainable_variables
mlayer_regularization_losses
nlayer_metrics
 

0
1
 

0
1
)2
*3
­
Btrainable_variables

olayers
Cregularization_losses
pmetrics
D	variables
qnon_trainable_variables
rlayer_regularization_losses
slayer_metrics
 

0
1
 

0
1
+2
,3
­
Gtrainable_variables

tlayers
Hregularization_losses
umetrics
I	variables
vnon_trainable_variables
wlayer_regularization_losses
xlayer_metrics
 

0
 

0
­
Ltrainable_variables

ylayers
Mregularization_losses
zmetrics
N	variables
{non_trainable_variables
|layer_regularization_losses
}layer_metrics

0
 

0
°
Ptrainable_variables

~layers
Qregularization_losses
metrics
R	variables
non_trainable_variables
 layer_regularization_losses
layer_metrics

0
 

0
²
Ttrainable_variables
layers
Uregularization_losses
metrics
V	variables
non_trainable_variables
 layer_regularization_losses
layer_metrics

 0
 

 0
²
Xtrainable_variables
layers
Yregularization_losses
metrics
Z	variables
non_trainable_variables
 layer_regularization_losses
layer_metrics

!0
"1
 

!0
"1
²
\trainable_variables
layers
]regularization_losses
metrics
^	variables
non_trainable_variables
 layer_regularization_losses
layer_metrics
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

'0
(1
 
 
 
 

)0
*1
 
 
 
 

+0
,1
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
|
serving_default_input_1Placeholder*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿÈ
³
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1Pnonshared_model_1/feed_forward_sub_net_10/batch_normalization_60/moving_varianceFnonshared_model_1/feed_forward_sub_net_10/batch_normalization_60/gammaLnonshared_model_1/feed_forward_sub_net_10/batch_normalization_60/moving_meanEnonshared_model_1/feed_forward_sub_net_10/batch_normalization_60/beta9nonshared_model_1/feed_forward_sub_net_10/dense_50/kernelPnonshared_model_1/feed_forward_sub_net_10/batch_normalization_61/moving_varianceFnonshared_model_1/feed_forward_sub_net_10/batch_normalization_61/gammaLnonshared_model_1/feed_forward_sub_net_10/batch_normalization_61/moving_meanEnonshared_model_1/feed_forward_sub_net_10/batch_normalization_61/beta9nonshared_model_1/feed_forward_sub_net_10/dense_51/kernelPnonshared_model_1/feed_forward_sub_net_10/batch_normalization_62/moving_varianceFnonshared_model_1/feed_forward_sub_net_10/batch_normalization_62/gammaLnonshared_model_1/feed_forward_sub_net_10/batch_normalization_62/moving_meanEnonshared_model_1/feed_forward_sub_net_10/batch_normalization_62/beta9nonshared_model_1/feed_forward_sub_net_10/dense_52/kernelPnonshared_model_1/feed_forward_sub_net_10/batch_normalization_63/moving_varianceFnonshared_model_1/feed_forward_sub_net_10/batch_normalization_63/gammaLnonshared_model_1/feed_forward_sub_net_10/batch_normalization_63/moving_meanEnonshared_model_1/feed_forward_sub_net_10/batch_normalization_63/beta9nonshared_model_1/feed_forward_sub_net_10/dense_53/kernelPnonshared_model_1/feed_forward_sub_net_10/batch_normalization_64/moving_varianceFnonshared_model_1/feed_forward_sub_net_10/batch_normalization_64/gammaLnonshared_model_1/feed_forward_sub_net_10/batch_normalization_64/moving_meanEnonshared_model_1/feed_forward_sub_net_10/batch_normalization_64/beta9nonshared_model_1/feed_forward_sub_net_10/dense_54/kernel7nonshared_model_1/feed_forward_sub_net_10/dense_54/bias*&
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_7049154
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameZnonshared_model_1/feed_forward_sub_net_10/batch_normalization_60/gamma/Read/ReadVariableOpYnonshared_model_1/feed_forward_sub_net_10/batch_normalization_60/beta/Read/ReadVariableOpZnonshared_model_1/feed_forward_sub_net_10/batch_normalization_61/gamma/Read/ReadVariableOpYnonshared_model_1/feed_forward_sub_net_10/batch_normalization_61/beta/Read/ReadVariableOpZnonshared_model_1/feed_forward_sub_net_10/batch_normalization_62/gamma/Read/ReadVariableOpYnonshared_model_1/feed_forward_sub_net_10/batch_normalization_62/beta/Read/ReadVariableOpZnonshared_model_1/feed_forward_sub_net_10/batch_normalization_63/gamma/Read/ReadVariableOpYnonshared_model_1/feed_forward_sub_net_10/batch_normalization_63/beta/Read/ReadVariableOpZnonshared_model_1/feed_forward_sub_net_10/batch_normalization_64/gamma/Read/ReadVariableOpYnonshared_model_1/feed_forward_sub_net_10/batch_normalization_64/beta/Read/ReadVariableOpMnonshared_model_1/feed_forward_sub_net_10/dense_50/kernel/Read/ReadVariableOpMnonshared_model_1/feed_forward_sub_net_10/dense_51/kernel/Read/ReadVariableOpMnonshared_model_1/feed_forward_sub_net_10/dense_52/kernel/Read/ReadVariableOpMnonshared_model_1/feed_forward_sub_net_10/dense_53/kernel/Read/ReadVariableOpMnonshared_model_1/feed_forward_sub_net_10/dense_54/kernel/Read/ReadVariableOpKnonshared_model_1/feed_forward_sub_net_10/dense_54/bias/Read/ReadVariableOp`nonshared_model_1/feed_forward_sub_net_10/batch_normalization_60/moving_mean/Read/ReadVariableOpdnonshared_model_1/feed_forward_sub_net_10/batch_normalization_60/moving_variance/Read/ReadVariableOp`nonshared_model_1/feed_forward_sub_net_10/batch_normalization_61/moving_mean/Read/ReadVariableOpdnonshared_model_1/feed_forward_sub_net_10/batch_normalization_61/moving_variance/Read/ReadVariableOp`nonshared_model_1/feed_forward_sub_net_10/batch_normalization_62/moving_mean/Read/ReadVariableOpdnonshared_model_1/feed_forward_sub_net_10/batch_normalization_62/moving_variance/Read/ReadVariableOp`nonshared_model_1/feed_forward_sub_net_10/batch_normalization_63/moving_mean/Read/ReadVariableOpdnonshared_model_1/feed_forward_sub_net_10/batch_normalization_63/moving_variance/Read/ReadVariableOp`nonshared_model_1/feed_forward_sub_net_10/batch_normalization_64/moving_mean/Read/ReadVariableOpdnonshared_model_1/feed_forward_sub_net_10/batch_normalization_64/moving_variance/Read/ReadVariableOpConst*'
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
GPU 2J 8 *)
f$R"
 __inference__traced_save_7050552
ü
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameFnonshared_model_1/feed_forward_sub_net_10/batch_normalization_60/gammaEnonshared_model_1/feed_forward_sub_net_10/batch_normalization_60/betaFnonshared_model_1/feed_forward_sub_net_10/batch_normalization_61/gammaEnonshared_model_1/feed_forward_sub_net_10/batch_normalization_61/betaFnonshared_model_1/feed_forward_sub_net_10/batch_normalization_62/gammaEnonshared_model_1/feed_forward_sub_net_10/batch_normalization_62/betaFnonshared_model_1/feed_forward_sub_net_10/batch_normalization_63/gammaEnonshared_model_1/feed_forward_sub_net_10/batch_normalization_63/betaFnonshared_model_1/feed_forward_sub_net_10/batch_normalization_64/gammaEnonshared_model_1/feed_forward_sub_net_10/batch_normalization_64/beta9nonshared_model_1/feed_forward_sub_net_10/dense_50/kernel9nonshared_model_1/feed_forward_sub_net_10/dense_51/kernel9nonshared_model_1/feed_forward_sub_net_10/dense_52/kernel9nonshared_model_1/feed_forward_sub_net_10/dense_53/kernel9nonshared_model_1/feed_forward_sub_net_10/dense_54/kernel7nonshared_model_1/feed_forward_sub_net_10/dense_54/biasLnonshared_model_1/feed_forward_sub_net_10/batch_normalization_60/moving_meanPnonshared_model_1/feed_forward_sub_net_10/batch_normalization_60/moving_varianceLnonshared_model_1/feed_forward_sub_net_10/batch_normalization_61/moving_meanPnonshared_model_1/feed_forward_sub_net_10/batch_normalization_61/moving_varianceLnonshared_model_1/feed_forward_sub_net_10/batch_normalization_62/moving_meanPnonshared_model_1/feed_forward_sub_net_10/batch_normalization_62/moving_varianceLnonshared_model_1/feed_forward_sub_net_10/batch_normalization_63/moving_meanPnonshared_model_1/feed_forward_sub_net_10/batch_normalization_63/moving_varianceLnonshared_model_1/feed_forward_sub_net_10/batch_normalization_64/moving_meanPnonshared_model_1/feed_forward_sub_net_10/batch_normalization_64/moving_variance*&
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
GPU 2J 8 *,
f'R%
#__inference__traced_restore_7050640õ»
ã
×
8__inference_batch_normalization_63_layer_call_fn_7050281

inputs
unknown:	Ü
	unknown_0:	Ü
	unknown_1:	Ü
	unknown_2:	Ü
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_63_layer_call_and_return_conditional_losses_70481922
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÜ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
 
_user_specified_nameinputs
ú

*__inference_dense_54_layer_call_fn_7050451

inputs
unknown:
ÜÈ
	unknown_0:	È
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_54_layer_call_and_return_conditional_losses_70486102
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÜ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
 
_user_specified_nameinputs
ö,
ð
S__inference_batch_normalization_61_layer_call_and_return_conditional_losses_7050104

inputs6
'assignmovingavg_readvariableop_resource:	Ü8
)assignmovingavg_1_readvariableop_resource:	Ü4
%batchnorm_mul_readvariableop_resource:	Ü0
!batchnorm_readvariableop_resource:	Ü
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	Ü*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	Ü2
moments/StopGradient¥
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices³
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	Ü*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:Ü*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:Ü*
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
AssignMovingAvg/Cast¥
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:Ü*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:Ü2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes	
:Ü2
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
AssignMovingAvg_1/Cast«
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:Ü*
dtype02"
 AssignMovingAvg_1/ReadVariableOp¡
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:Ü2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes	
:Ü2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1k
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Ü2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ü2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2

Identityò
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÜ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
 
_user_specified_nameinputs
ö,
ð
S__inference_batch_normalization_61_layer_call_and_return_conditional_losses_7047922

inputs6
'assignmovingavg_readvariableop_resource:	Ü8
)assignmovingavg_1_readvariableop_resource:	Ü4
%batchnorm_mul_readvariableop_resource:	Ü0
!batchnorm_readvariableop_resource:	Ü
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	Ü*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	Ü2
moments/StopGradient¥
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices³
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	Ü*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:Ü*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:Ü*
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
AssignMovingAvg/Cast¥
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:Ü*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:Ü2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes	
:Ü2
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
AssignMovingAvg_1/Cast«
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:Ü*
dtype02"
 AssignMovingAvg_1/ReadVariableOp¡
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:Ü2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes	
:Ü2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1k
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Ü2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ü2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2

Identityò
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÜ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
 
_user_specified_nameinputs
á
×
8__inference_batch_normalization_62_layer_call_fn_7050212

inputs
unknown:	Ü
	unknown_0:	Ü
	unknown_1:	Ü
	unknown_2:	Ü
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_62_layer_call_and_return_conditional_losses_70480882
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÜ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
 
_user_specified_nameinputs

°
E__inference_dense_52_layer_call_and_return_conditional_losses_7050411

inputs2
matmul_readvariableop_resource:
ÜÜ
identity¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ÜÜ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
MatMull
IdentityIdentityMatMul:product:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:ÿÿÿÿÿÿÿÿÿÜ: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
 
_user_specified_nameinputs
ï
à
#__inference__traced_restore_7050640
file_prefixf
Wassignvariableop_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_60_gamma:	Èg
Xassignvariableop_1_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_60_beta:	Èh
Yassignvariableop_2_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_61_gamma:	Üg
Xassignvariableop_3_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_61_beta:	Üh
Yassignvariableop_4_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_62_gamma:	Üg
Xassignvariableop_5_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_62_beta:	Üh
Yassignvariableop_6_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_63_gamma:	Üg
Xassignvariableop_7_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_63_beta:	Üh
Yassignvariableop_8_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_64_gamma:	Üg
Xassignvariableop_9_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_64_beta:	Üa
Massignvariableop_10_nonshared_model_1_feed_forward_sub_net_10_dense_50_kernel:
ÈÜa
Massignvariableop_11_nonshared_model_1_feed_forward_sub_net_10_dense_51_kernel:
ÜÜa
Massignvariableop_12_nonshared_model_1_feed_forward_sub_net_10_dense_52_kernel:
ÜÜa
Massignvariableop_13_nonshared_model_1_feed_forward_sub_net_10_dense_53_kernel:
ÜÜa
Massignvariableop_14_nonshared_model_1_feed_forward_sub_net_10_dense_54_kernel:
ÜÈZ
Kassignvariableop_15_nonshared_model_1_feed_forward_sub_net_10_dense_54_bias:	Èo
`assignvariableop_16_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_60_moving_mean:	Ès
dassignvariableop_17_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_60_moving_variance:	Èo
`assignvariableop_18_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_61_moving_mean:	Üs
dassignvariableop_19_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_61_moving_variance:	Üo
`assignvariableop_20_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_62_moving_mean:	Üs
dassignvariableop_21_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_62_moving_variance:	Üo
`assignvariableop_22_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_63_moving_mean:	Üs
dassignvariableop_23_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_63_moving_variance:	Üo
`assignvariableop_24_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_64_moving_mean:	Üs
dassignvariableop_25_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_64_moving_variance:	Ü
identity_27¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9ç

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ó	
valueé	Bæ	B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
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

IdentityÖ
AssignVariableOpAssignVariableOpWassignvariableop_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_60_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ý
AssignVariableOp_1AssignVariableOpXassignvariableop_1_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_60_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Þ
AssignVariableOp_2AssignVariableOpYassignvariableop_2_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_61_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ý
AssignVariableOp_3AssignVariableOpXassignvariableop_3_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_61_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Þ
AssignVariableOp_4AssignVariableOpYassignvariableop_4_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_62_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ý
AssignVariableOp_5AssignVariableOpXassignvariableop_5_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_62_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Þ
AssignVariableOp_6AssignVariableOpYassignvariableop_6_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_63_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ý
AssignVariableOp_7AssignVariableOpXassignvariableop_7_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_63_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Þ
AssignVariableOp_8AssignVariableOpYassignvariableop_8_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_64_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ý
AssignVariableOp_9AssignVariableOpXassignvariableop_9_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_64_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Õ
AssignVariableOp_10AssignVariableOpMassignvariableop_10_nonshared_model_1_feed_forward_sub_net_10_dense_50_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Õ
AssignVariableOp_11AssignVariableOpMassignvariableop_11_nonshared_model_1_feed_forward_sub_net_10_dense_51_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Õ
AssignVariableOp_12AssignVariableOpMassignvariableop_12_nonshared_model_1_feed_forward_sub_net_10_dense_52_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Õ
AssignVariableOp_13AssignVariableOpMassignvariableop_13_nonshared_model_1_feed_forward_sub_net_10_dense_53_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Õ
AssignVariableOp_14AssignVariableOpMassignvariableop_14_nonshared_model_1_feed_forward_sub_net_10_dense_54_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Ó
AssignVariableOp_15AssignVariableOpKassignvariableop_15_nonshared_model_1_feed_forward_sub_net_10_dense_54_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16è
AssignVariableOp_16AssignVariableOp`assignvariableop_16_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_60_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17ì
AssignVariableOp_17AssignVariableOpdassignvariableop_17_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_60_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18è
AssignVariableOp_18AssignVariableOp`assignvariableop_18_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_61_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19ì
AssignVariableOp_19AssignVariableOpdassignvariableop_19_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_61_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20è
AssignVariableOp_20AssignVariableOp`assignvariableop_20_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_62_moving_meanIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21ì
AssignVariableOp_21AssignVariableOpdassignvariableop_21_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_62_moving_varianceIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22è
AssignVariableOp_22AssignVariableOp`assignvariableop_22_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_63_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23ì
AssignVariableOp_23AssignVariableOpdassignvariableop_23_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_63_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24è
AssignVariableOp_24AssignVariableOp`assignvariableop_24_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_64_moving_meanIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25ì
AssignVariableOp_25AssignVariableOpdassignvariableop_25_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_64_moving_varianceIdentity_25:output:0"/device:CPU:0*
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

¶
S__inference_batch_normalization_60_layer_call_and_return_conditional_losses_7047694

inputs0
!batchnorm_readvariableop_resource:	È4
%batchnorm_mul_readvariableop_resource:	È2
#batchnorm_readvariableop_1_resource:	È2
#batchnorm_readvariableop_2_resource:	È
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:È*
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:È2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:È2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:È*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:È2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:È*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:È2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:È*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:È2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

IdentityÂ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
ß
©
%__inference_signature_wrapper_7049154
input_1
unknown:	È
	unknown_0:	È
	unknown_1:	È
	unknown_2:	È
	unknown_3:
ÈÜ
	unknown_4:	Ü
	unknown_5:	Ü
	unknown_6:	Ü
	unknown_7:	Ü
	unknown_8:
ÜÜ
	unknown_9:	Ü

unknown_10:	Ü

unknown_11:	Ü

unknown_12:	Ü

unknown_13:
ÜÜ

unknown_14:	Ü

unknown_15:	Ü

unknown_16:	Ü

unknown_17:	Ü

unknown_18:
ÜÜ

unknown_19:	Ü

unknown_20:	Ü

unknown_21:	Ü

unknown_22:	Ü

unknown_23:
ÜÈ

unknown_24:	È
identity¢StatefulPartitionedCall
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
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_70476702
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:ÿÿÿÿÿÿÿÿÿÈ: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
!
_user_specified_name	input_1

¶
S__inference_batch_normalization_62_layer_call_and_return_conditional_losses_7048026

inputs0
!batchnorm_readvariableop_resource:	Ü4
%batchnorm_mul_readvariableop_resource:	Ü2
#batchnorm_readvariableop_1_resource:	Ü2
#batchnorm_readvariableop_2_resource:	Ü
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Ü2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ü2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2

IdentityÂ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÜ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
 
_user_specified_nameinputs
Ñ

*__inference_dense_52_layer_call_fn_7050418

inputs
unknown:
ÜÜ
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_52_layer_call_and_return_conditional_losses_70485652
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:ÿÿÿÿÿÿÿÿÿÜ: 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
 
_user_specified_nameinputs

¶
S__inference_batch_normalization_61_layer_call_and_return_conditional_losses_7050068

inputs0
!batchnorm_readvariableop_resource:	Ü4
%batchnorm_mul_readvariableop_resource:	Ü2
#batchnorm_readvariableop_1_resource:	Ü2
#batchnorm_readvariableop_2_resource:	Ü
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Ü2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ü2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2

IdentityÂ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÜ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
 
_user_specified_nameinputs
ÎE

T__inference_feed_forward_sub_net_10_layer_call_and_return_conditional_losses_7048617
x-
batch_normalization_60_7048507:	È-
batch_normalization_60_7048509:	È-
batch_normalization_60_7048511:	È-
batch_normalization_60_7048513:	È$
dense_50_7048524:
ÈÜ-
batch_normalization_61_7048527:	Ü-
batch_normalization_61_7048529:	Ü-
batch_normalization_61_7048531:	Ü-
batch_normalization_61_7048533:	Ü$
dense_51_7048545:
ÜÜ-
batch_normalization_62_7048548:	Ü-
batch_normalization_62_7048550:	Ü-
batch_normalization_62_7048552:	Ü-
batch_normalization_62_7048554:	Ü$
dense_52_7048566:
ÜÜ-
batch_normalization_63_7048569:	Ü-
batch_normalization_63_7048571:	Ü-
batch_normalization_63_7048573:	Ü-
batch_normalization_63_7048575:	Ü$
dense_53_7048587:
ÜÜ-
batch_normalization_64_7048590:	Ü-
batch_normalization_64_7048592:	Ü-
batch_normalization_64_7048594:	Ü-
batch_normalization_64_7048596:	Ü$
dense_54_7048611:
ÜÈ
dense_54_7048613:	È
identity¢.batch_normalization_60/StatefulPartitionedCall¢.batch_normalization_61/StatefulPartitionedCall¢.batch_normalization_62/StatefulPartitionedCall¢.batch_normalization_63/StatefulPartitionedCall¢.batch_normalization_64/StatefulPartitionedCall¢ dense_50/StatefulPartitionedCall¢ dense_51/StatefulPartitionedCall¢ dense_52/StatefulPartitionedCall¢ dense_53/StatefulPartitionedCall¢ dense_54/StatefulPartitionedCall
.batch_normalization_60/StatefulPartitionedCallStatefulPartitionedCallxbatch_normalization_60_7048507batch_normalization_60_7048509batch_normalization_60_7048511batch_normalization_60_7048513*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_60_layer_call_and_return_conditional_losses_704769420
.batch_normalization_60/StatefulPartitionedCallµ
 dense_50/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_60/StatefulPartitionedCall:output:0dense_50_7048524*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_50_layer_call_and_return_conditional_losses_70485232"
 dense_50/StatefulPartitionedCallÅ
.batch_normalization_61/StatefulPartitionedCallStatefulPartitionedCall)dense_50/StatefulPartitionedCall:output:0batch_normalization_61_7048527batch_normalization_61_7048529batch_normalization_61_7048531batch_normalization_61_7048533*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_61_layer_call_and_return_conditional_losses_704786020
.batch_normalization_61/StatefulPartitionedCall
ReluRelu7batch_normalization_61/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
Relu
 dense_51/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0dense_51_7048545*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_51_layer_call_and_return_conditional_losses_70485442"
 dense_51/StatefulPartitionedCallÅ
.batch_normalization_62/StatefulPartitionedCallStatefulPartitionedCall)dense_51/StatefulPartitionedCall:output:0batch_normalization_62_7048548batch_normalization_62_7048550batch_normalization_62_7048552batch_normalization_62_7048554*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_62_layer_call_and_return_conditional_losses_704802620
.batch_normalization_62/StatefulPartitionedCall
Relu_1Relu7batch_normalization_62/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
Relu_1
 dense_52/StatefulPartitionedCallStatefulPartitionedCallRelu_1:activations:0dense_52_7048566*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_52_layer_call_and_return_conditional_losses_70485652"
 dense_52/StatefulPartitionedCallÅ
.batch_normalization_63/StatefulPartitionedCallStatefulPartitionedCall)dense_52/StatefulPartitionedCall:output:0batch_normalization_63_7048569batch_normalization_63_7048571batch_normalization_63_7048573batch_normalization_63_7048575*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_63_layer_call_and_return_conditional_losses_704819220
.batch_normalization_63/StatefulPartitionedCall
Relu_2Relu7batch_normalization_63/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
Relu_2
 dense_53/StatefulPartitionedCallStatefulPartitionedCallRelu_2:activations:0dense_53_7048587*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_53_layer_call_and_return_conditional_losses_70485862"
 dense_53/StatefulPartitionedCallÅ
.batch_normalization_64/StatefulPartitionedCallStatefulPartitionedCall)dense_53/StatefulPartitionedCall:output:0batch_normalization_64_7048590batch_normalization_64_7048592batch_normalization_64_7048594batch_normalization_64_7048596*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_64_layer_call_and_return_conditional_losses_704835820
.batch_normalization_64/StatefulPartitionedCall
Relu_3Relu7batch_normalization_64/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
Relu_3¦
 dense_54/StatefulPartitionedCallStatefulPartitionedCallRelu_3:activations:0dense_54_7048611dense_54_7048613*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_54_layer_call_and_return_conditional_losses_70486102"
 dense_54/StatefulPartitionedCall
IdentityIdentity)dense_54/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identityò
NoOpNoOp/^batch_normalization_60/StatefulPartitionedCall/^batch_normalization_61/StatefulPartitionedCall/^batch_normalization_62/StatefulPartitionedCall/^batch_normalization_63/StatefulPartitionedCall/^batch_normalization_64/StatefulPartitionedCall!^dense_50/StatefulPartitionedCall!^dense_51/StatefulPartitionedCall!^dense_52/StatefulPartitionedCall!^dense_53/StatefulPartitionedCall!^dense_54/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:ÿÿÿÿÿÿÿÿÿÈ: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_60/StatefulPartitionedCall.batch_normalization_60/StatefulPartitionedCall2`
.batch_normalization_61/StatefulPartitionedCall.batch_normalization_61/StatefulPartitionedCall2`
.batch_normalization_62/StatefulPartitionedCall.batch_normalization_62/StatefulPartitionedCall2`
.batch_normalization_63/StatefulPartitionedCall.batch_normalization_63/StatefulPartitionedCall2`
.batch_normalization_64/StatefulPartitionedCall.batch_normalization_64/StatefulPartitionedCall2D
 dense_50/StatefulPartitionedCall dense_50/StatefulPartitionedCall2D
 dense_51/StatefulPartitionedCall dense_51/StatefulPartitionedCall2D
 dense_52/StatefulPartitionedCall dense_52/StatefulPartitionedCall2D
 dense_53/StatefulPartitionedCall dense_53/StatefulPartitionedCall2D
 dense_54/StatefulPartitionedCall dense_54/StatefulPartitionedCall:K G
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ

_user_specified_namex
÷±

T__inference_feed_forward_sub_net_10_layer_call_and_return_conditional_losses_7049260
xG
8batch_normalization_60_batchnorm_readvariableop_resource:	ÈK
<batch_normalization_60_batchnorm_mul_readvariableop_resource:	ÈI
:batch_normalization_60_batchnorm_readvariableop_1_resource:	ÈI
:batch_normalization_60_batchnorm_readvariableop_2_resource:	È;
'dense_50_matmul_readvariableop_resource:
ÈÜG
8batch_normalization_61_batchnorm_readvariableop_resource:	ÜK
<batch_normalization_61_batchnorm_mul_readvariableop_resource:	ÜI
:batch_normalization_61_batchnorm_readvariableop_1_resource:	ÜI
:batch_normalization_61_batchnorm_readvariableop_2_resource:	Ü;
'dense_51_matmul_readvariableop_resource:
ÜÜG
8batch_normalization_62_batchnorm_readvariableop_resource:	ÜK
<batch_normalization_62_batchnorm_mul_readvariableop_resource:	ÜI
:batch_normalization_62_batchnorm_readvariableop_1_resource:	ÜI
:batch_normalization_62_batchnorm_readvariableop_2_resource:	Ü;
'dense_52_matmul_readvariableop_resource:
ÜÜG
8batch_normalization_63_batchnorm_readvariableop_resource:	ÜK
<batch_normalization_63_batchnorm_mul_readvariableop_resource:	ÜI
:batch_normalization_63_batchnorm_readvariableop_1_resource:	ÜI
:batch_normalization_63_batchnorm_readvariableop_2_resource:	Ü;
'dense_53_matmul_readvariableop_resource:
ÜÜG
8batch_normalization_64_batchnorm_readvariableop_resource:	ÜK
<batch_normalization_64_batchnorm_mul_readvariableop_resource:	ÜI
:batch_normalization_64_batchnorm_readvariableop_1_resource:	ÜI
:batch_normalization_64_batchnorm_readvariableop_2_resource:	Ü;
'dense_54_matmul_readvariableop_resource:
ÜÈ7
(dense_54_biasadd_readvariableop_resource:	È
identity¢/batch_normalization_60/batchnorm/ReadVariableOp¢1batch_normalization_60/batchnorm/ReadVariableOp_1¢1batch_normalization_60/batchnorm/ReadVariableOp_2¢3batch_normalization_60/batchnorm/mul/ReadVariableOp¢/batch_normalization_61/batchnorm/ReadVariableOp¢1batch_normalization_61/batchnorm/ReadVariableOp_1¢1batch_normalization_61/batchnorm/ReadVariableOp_2¢3batch_normalization_61/batchnorm/mul/ReadVariableOp¢/batch_normalization_62/batchnorm/ReadVariableOp¢1batch_normalization_62/batchnorm/ReadVariableOp_1¢1batch_normalization_62/batchnorm/ReadVariableOp_2¢3batch_normalization_62/batchnorm/mul/ReadVariableOp¢/batch_normalization_63/batchnorm/ReadVariableOp¢1batch_normalization_63/batchnorm/ReadVariableOp_1¢1batch_normalization_63/batchnorm/ReadVariableOp_2¢3batch_normalization_63/batchnorm/mul/ReadVariableOp¢/batch_normalization_64/batchnorm/ReadVariableOp¢1batch_normalization_64/batchnorm/ReadVariableOp_1¢1batch_normalization_64/batchnorm/ReadVariableOp_2¢3batch_normalization_64/batchnorm/mul/ReadVariableOp¢dense_50/MatMul/ReadVariableOp¢dense_51/MatMul/ReadVariableOp¢dense_52/MatMul/ReadVariableOp¢dense_53/MatMul/ReadVariableOp¢dense_54/BiasAdd/ReadVariableOp¢dense_54/MatMul/ReadVariableOpØ
/batch_normalization_60/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_60_batchnorm_readvariableop_resource*
_output_shapes	
:È*
dtype021
/batch_normalization_60/batchnorm/ReadVariableOp
&batch_normalization_60/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_60/batchnorm/add/yå
$batch_normalization_60/batchnorm/addAddV27batch_normalization_60/batchnorm/ReadVariableOp:value:0/batch_normalization_60/batchnorm/add/y:output:0*
T0*
_output_shapes	
:È2&
$batch_normalization_60/batchnorm/add©
&batch_normalization_60/batchnorm/RsqrtRsqrt(batch_normalization_60/batchnorm/add:z:0*
T0*
_output_shapes	
:È2(
&batch_normalization_60/batchnorm/Rsqrtä
3batch_normalization_60/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_60_batchnorm_mul_readvariableop_resource*
_output_shapes	
:È*
dtype025
3batch_normalization_60/batchnorm/mul/ReadVariableOpâ
$batch_normalization_60/batchnorm/mulMul*batch_normalization_60/batchnorm/Rsqrt:y:0;batch_normalization_60/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:È2&
$batch_normalization_60/batchnorm/mul·
&batch_normalization_60/batchnorm/mul_1Mulx(batch_normalization_60/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2(
&batch_normalization_60/batchnorm/mul_1Þ
1batch_normalization_60/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_60_batchnorm_readvariableop_1_resource*
_output_shapes	
:È*
dtype023
1batch_normalization_60/batchnorm/ReadVariableOp_1â
&batch_normalization_60/batchnorm/mul_2Mul9batch_normalization_60/batchnorm/ReadVariableOp_1:value:0(batch_normalization_60/batchnorm/mul:z:0*
T0*
_output_shapes	
:È2(
&batch_normalization_60/batchnorm/mul_2Þ
1batch_normalization_60/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_60_batchnorm_readvariableop_2_resource*
_output_shapes	
:È*
dtype023
1batch_normalization_60/batchnorm/ReadVariableOp_2à
$batch_normalization_60/batchnorm/subSub9batch_normalization_60/batchnorm/ReadVariableOp_2:value:0*batch_normalization_60/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:È2&
$batch_normalization_60/batchnorm/subâ
&batch_normalization_60/batchnorm/add_1AddV2*batch_normalization_60/batchnorm/mul_1:z:0(batch_normalization_60/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2(
&batch_normalization_60/batchnorm/add_1ª
dense_50/MatMul/ReadVariableOpReadVariableOp'dense_50_matmul_readvariableop_resource* 
_output_shapes
:
ÈÜ*
dtype02 
dense_50/MatMul/ReadVariableOp³
dense_50/MatMulMatMul*batch_normalization_60/batchnorm/add_1:z:0&dense_50/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
dense_50/MatMulØ
/batch_normalization_61/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_61_batchnorm_readvariableop_resource*
_output_shapes	
:Ü*
dtype021
/batch_normalization_61/batchnorm/ReadVariableOp
&batch_normalization_61/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_61/batchnorm/add/yå
$batch_normalization_61/batchnorm/addAddV27batch_normalization_61/batchnorm/ReadVariableOp:value:0/batch_normalization_61/batchnorm/add/y:output:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_61/batchnorm/add©
&batch_normalization_61/batchnorm/RsqrtRsqrt(batch_normalization_61/batchnorm/add:z:0*
T0*
_output_shapes	
:Ü2(
&batch_normalization_61/batchnorm/Rsqrtä
3batch_normalization_61/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_61_batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ü*
dtype025
3batch_normalization_61/batchnorm/mul/ReadVariableOpâ
$batch_normalization_61/batchnorm/mulMul*batch_normalization_61/batchnorm/Rsqrt:y:0;batch_normalization_61/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_61/batchnorm/mulÏ
&batch_normalization_61/batchnorm/mul_1Muldense_50/MatMul:product:0(batch_normalization_61/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2(
&batch_normalization_61/batchnorm/mul_1Þ
1batch_normalization_61/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_61_batchnorm_readvariableop_1_resource*
_output_shapes	
:Ü*
dtype023
1batch_normalization_61/batchnorm/ReadVariableOp_1â
&batch_normalization_61/batchnorm/mul_2Mul9batch_normalization_61/batchnorm/ReadVariableOp_1:value:0(batch_normalization_61/batchnorm/mul:z:0*
T0*
_output_shapes	
:Ü2(
&batch_normalization_61/batchnorm/mul_2Þ
1batch_normalization_61/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_61_batchnorm_readvariableop_2_resource*
_output_shapes	
:Ü*
dtype023
1batch_normalization_61/batchnorm/ReadVariableOp_2à
$batch_normalization_61/batchnorm/subSub9batch_normalization_61/batchnorm/ReadVariableOp_2:value:0*batch_normalization_61/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_61/batchnorm/subâ
&batch_normalization_61/batchnorm/add_1AddV2*batch_normalization_61/batchnorm/mul_1:z:0(batch_normalization_61/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2(
&batch_normalization_61/batchnorm/add_1s
ReluRelu*batch_normalization_61/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
Reluª
dense_51/MatMul/ReadVariableOpReadVariableOp'dense_51_matmul_readvariableop_resource* 
_output_shapes
:
ÜÜ*
dtype02 
dense_51/MatMul/ReadVariableOp
dense_51/MatMulMatMulRelu:activations:0&dense_51/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
dense_51/MatMulØ
/batch_normalization_62/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_62_batchnorm_readvariableop_resource*
_output_shapes	
:Ü*
dtype021
/batch_normalization_62/batchnorm/ReadVariableOp
&batch_normalization_62/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_62/batchnorm/add/yå
$batch_normalization_62/batchnorm/addAddV27batch_normalization_62/batchnorm/ReadVariableOp:value:0/batch_normalization_62/batchnorm/add/y:output:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_62/batchnorm/add©
&batch_normalization_62/batchnorm/RsqrtRsqrt(batch_normalization_62/batchnorm/add:z:0*
T0*
_output_shapes	
:Ü2(
&batch_normalization_62/batchnorm/Rsqrtä
3batch_normalization_62/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_62_batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ü*
dtype025
3batch_normalization_62/batchnorm/mul/ReadVariableOpâ
$batch_normalization_62/batchnorm/mulMul*batch_normalization_62/batchnorm/Rsqrt:y:0;batch_normalization_62/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_62/batchnorm/mulÏ
&batch_normalization_62/batchnorm/mul_1Muldense_51/MatMul:product:0(batch_normalization_62/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2(
&batch_normalization_62/batchnorm/mul_1Þ
1batch_normalization_62/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_62_batchnorm_readvariableop_1_resource*
_output_shapes	
:Ü*
dtype023
1batch_normalization_62/batchnorm/ReadVariableOp_1â
&batch_normalization_62/batchnorm/mul_2Mul9batch_normalization_62/batchnorm/ReadVariableOp_1:value:0(batch_normalization_62/batchnorm/mul:z:0*
T0*
_output_shapes	
:Ü2(
&batch_normalization_62/batchnorm/mul_2Þ
1batch_normalization_62/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_62_batchnorm_readvariableop_2_resource*
_output_shapes	
:Ü*
dtype023
1batch_normalization_62/batchnorm/ReadVariableOp_2à
$batch_normalization_62/batchnorm/subSub9batch_normalization_62/batchnorm/ReadVariableOp_2:value:0*batch_normalization_62/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_62/batchnorm/subâ
&batch_normalization_62/batchnorm/add_1AddV2*batch_normalization_62/batchnorm/mul_1:z:0(batch_normalization_62/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2(
&batch_normalization_62/batchnorm/add_1w
Relu_1Relu*batch_normalization_62/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
Relu_1ª
dense_52/MatMul/ReadVariableOpReadVariableOp'dense_52_matmul_readvariableop_resource* 
_output_shapes
:
ÜÜ*
dtype02 
dense_52/MatMul/ReadVariableOp
dense_52/MatMulMatMulRelu_1:activations:0&dense_52/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
dense_52/MatMulØ
/batch_normalization_63/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_63_batchnorm_readvariableop_resource*
_output_shapes	
:Ü*
dtype021
/batch_normalization_63/batchnorm/ReadVariableOp
&batch_normalization_63/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_63/batchnorm/add/yå
$batch_normalization_63/batchnorm/addAddV27batch_normalization_63/batchnorm/ReadVariableOp:value:0/batch_normalization_63/batchnorm/add/y:output:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_63/batchnorm/add©
&batch_normalization_63/batchnorm/RsqrtRsqrt(batch_normalization_63/batchnorm/add:z:0*
T0*
_output_shapes	
:Ü2(
&batch_normalization_63/batchnorm/Rsqrtä
3batch_normalization_63/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_63_batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ü*
dtype025
3batch_normalization_63/batchnorm/mul/ReadVariableOpâ
$batch_normalization_63/batchnorm/mulMul*batch_normalization_63/batchnorm/Rsqrt:y:0;batch_normalization_63/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_63/batchnorm/mulÏ
&batch_normalization_63/batchnorm/mul_1Muldense_52/MatMul:product:0(batch_normalization_63/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2(
&batch_normalization_63/batchnorm/mul_1Þ
1batch_normalization_63/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_63_batchnorm_readvariableop_1_resource*
_output_shapes	
:Ü*
dtype023
1batch_normalization_63/batchnorm/ReadVariableOp_1â
&batch_normalization_63/batchnorm/mul_2Mul9batch_normalization_63/batchnorm/ReadVariableOp_1:value:0(batch_normalization_63/batchnorm/mul:z:0*
T0*
_output_shapes	
:Ü2(
&batch_normalization_63/batchnorm/mul_2Þ
1batch_normalization_63/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_63_batchnorm_readvariableop_2_resource*
_output_shapes	
:Ü*
dtype023
1batch_normalization_63/batchnorm/ReadVariableOp_2à
$batch_normalization_63/batchnorm/subSub9batch_normalization_63/batchnorm/ReadVariableOp_2:value:0*batch_normalization_63/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_63/batchnorm/subâ
&batch_normalization_63/batchnorm/add_1AddV2*batch_normalization_63/batchnorm/mul_1:z:0(batch_normalization_63/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2(
&batch_normalization_63/batchnorm/add_1w
Relu_2Relu*batch_normalization_63/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
Relu_2ª
dense_53/MatMul/ReadVariableOpReadVariableOp'dense_53_matmul_readvariableop_resource* 
_output_shapes
:
ÜÜ*
dtype02 
dense_53/MatMul/ReadVariableOp
dense_53/MatMulMatMulRelu_2:activations:0&dense_53/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
dense_53/MatMulØ
/batch_normalization_64/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_64_batchnorm_readvariableop_resource*
_output_shapes	
:Ü*
dtype021
/batch_normalization_64/batchnorm/ReadVariableOp
&batch_normalization_64/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_64/batchnorm/add/yå
$batch_normalization_64/batchnorm/addAddV27batch_normalization_64/batchnorm/ReadVariableOp:value:0/batch_normalization_64/batchnorm/add/y:output:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_64/batchnorm/add©
&batch_normalization_64/batchnorm/RsqrtRsqrt(batch_normalization_64/batchnorm/add:z:0*
T0*
_output_shapes	
:Ü2(
&batch_normalization_64/batchnorm/Rsqrtä
3batch_normalization_64/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_64_batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ü*
dtype025
3batch_normalization_64/batchnorm/mul/ReadVariableOpâ
$batch_normalization_64/batchnorm/mulMul*batch_normalization_64/batchnorm/Rsqrt:y:0;batch_normalization_64/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_64/batchnorm/mulÏ
&batch_normalization_64/batchnorm/mul_1Muldense_53/MatMul:product:0(batch_normalization_64/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2(
&batch_normalization_64/batchnorm/mul_1Þ
1batch_normalization_64/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_64_batchnorm_readvariableop_1_resource*
_output_shapes	
:Ü*
dtype023
1batch_normalization_64/batchnorm/ReadVariableOp_1â
&batch_normalization_64/batchnorm/mul_2Mul9batch_normalization_64/batchnorm/ReadVariableOp_1:value:0(batch_normalization_64/batchnorm/mul:z:0*
T0*
_output_shapes	
:Ü2(
&batch_normalization_64/batchnorm/mul_2Þ
1batch_normalization_64/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_64_batchnorm_readvariableop_2_resource*
_output_shapes	
:Ü*
dtype023
1batch_normalization_64/batchnorm/ReadVariableOp_2à
$batch_normalization_64/batchnorm/subSub9batch_normalization_64/batchnorm/ReadVariableOp_2:value:0*batch_normalization_64/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_64/batchnorm/subâ
&batch_normalization_64/batchnorm/add_1AddV2*batch_normalization_64/batchnorm/mul_1:z:0(batch_normalization_64/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2(
&batch_normalization_64/batchnorm/add_1w
Relu_3Relu*batch_normalization_64/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
Relu_3ª
dense_54/MatMul/ReadVariableOpReadVariableOp'dense_54_matmul_readvariableop_resource* 
_output_shapes
:
ÜÈ*
dtype02 
dense_54/MatMul/ReadVariableOp
dense_54/MatMulMatMulRelu_3:activations:0&dense_54/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
dense_54/MatMul¨
dense_54/BiasAdd/ReadVariableOpReadVariableOp(dense_54_biasadd_readvariableop_resource*
_output_shapes	
:È*
dtype02!
dense_54/BiasAdd/ReadVariableOp¦
dense_54/BiasAddBiasAdddense_54/MatMul:product:0'dense_54/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
dense_54/BiasAddu
IdentityIdentitydense_54/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity¥

NoOpNoOp0^batch_normalization_60/batchnorm/ReadVariableOp2^batch_normalization_60/batchnorm/ReadVariableOp_12^batch_normalization_60/batchnorm/ReadVariableOp_24^batch_normalization_60/batchnorm/mul/ReadVariableOp0^batch_normalization_61/batchnorm/ReadVariableOp2^batch_normalization_61/batchnorm/ReadVariableOp_12^batch_normalization_61/batchnorm/ReadVariableOp_24^batch_normalization_61/batchnorm/mul/ReadVariableOp0^batch_normalization_62/batchnorm/ReadVariableOp2^batch_normalization_62/batchnorm/ReadVariableOp_12^batch_normalization_62/batchnorm/ReadVariableOp_24^batch_normalization_62/batchnorm/mul/ReadVariableOp0^batch_normalization_63/batchnorm/ReadVariableOp2^batch_normalization_63/batchnorm/ReadVariableOp_12^batch_normalization_63/batchnorm/ReadVariableOp_24^batch_normalization_63/batchnorm/mul/ReadVariableOp0^batch_normalization_64/batchnorm/ReadVariableOp2^batch_normalization_64/batchnorm/ReadVariableOp_12^batch_normalization_64/batchnorm/ReadVariableOp_24^batch_normalization_64/batchnorm/mul/ReadVariableOp^dense_50/MatMul/ReadVariableOp^dense_51/MatMul/ReadVariableOp^dense_52/MatMul/ReadVariableOp^dense_53/MatMul/ReadVariableOp ^dense_54/BiasAdd/ReadVariableOp^dense_54/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:ÿÿÿÿÿÿÿÿÿÈ: : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_60/batchnorm/ReadVariableOp/batch_normalization_60/batchnorm/ReadVariableOp2f
1batch_normalization_60/batchnorm/ReadVariableOp_11batch_normalization_60/batchnorm/ReadVariableOp_12f
1batch_normalization_60/batchnorm/ReadVariableOp_21batch_normalization_60/batchnorm/ReadVariableOp_22j
3batch_normalization_60/batchnorm/mul/ReadVariableOp3batch_normalization_60/batchnorm/mul/ReadVariableOp2b
/batch_normalization_61/batchnorm/ReadVariableOp/batch_normalization_61/batchnorm/ReadVariableOp2f
1batch_normalization_61/batchnorm/ReadVariableOp_11batch_normalization_61/batchnorm/ReadVariableOp_12f
1batch_normalization_61/batchnorm/ReadVariableOp_21batch_normalization_61/batchnorm/ReadVariableOp_22j
3batch_normalization_61/batchnorm/mul/ReadVariableOp3batch_normalization_61/batchnorm/mul/ReadVariableOp2b
/batch_normalization_62/batchnorm/ReadVariableOp/batch_normalization_62/batchnorm/ReadVariableOp2f
1batch_normalization_62/batchnorm/ReadVariableOp_11batch_normalization_62/batchnorm/ReadVariableOp_12f
1batch_normalization_62/batchnorm/ReadVariableOp_21batch_normalization_62/batchnorm/ReadVariableOp_22j
3batch_normalization_62/batchnorm/mul/ReadVariableOp3batch_normalization_62/batchnorm/mul/ReadVariableOp2b
/batch_normalization_63/batchnorm/ReadVariableOp/batch_normalization_63/batchnorm/ReadVariableOp2f
1batch_normalization_63/batchnorm/ReadVariableOp_11batch_normalization_63/batchnorm/ReadVariableOp_12f
1batch_normalization_63/batchnorm/ReadVariableOp_21batch_normalization_63/batchnorm/ReadVariableOp_22j
3batch_normalization_63/batchnorm/mul/ReadVariableOp3batch_normalization_63/batchnorm/mul/ReadVariableOp2b
/batch_normalization_64/batchnorm/ReadVariableOp/batch_normalization_64/batchnorm/ReadVariableOp2f
1batch_normalization_64/batchnorm/ReadVariableOp_11batch_normalization_64/batchnorm/ReadVariableOp_12f
1batch_normalization_64/batchnorm/ReadVariableOp_21batch_normalization_64/batchnorm/ReadVariableOp_22j
3batch_normalization_64/batchnorm/mul/ReadVariableOp3batch_normalization_64/batchnorm/mul/ReadVariableOp2@
dense_50/MatMul/ReadVariableOpdense_50/MatMul/ReadVariableOp2@
dense_51/MatMul/ReadVariableOpdense_51/MatMul/ReadVariableOp2@
dense_52/MatMul/ReadVariableOpdense_52/MatMul/ReadVariableOp2@
dense_53/MatMul/ReadVariableOpdense_53/MatMul/ReadVariableOp2B
dense_54/BiasAdd/ReadVariableOpdense_54/BiasAdd/ReadVariableOp2@
dense_54/MatMul/ReadVariableOpdense_54/MatMul/ReadVariableOp:K G
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ

_user_specified_namex

¶
S__inference_batch_normalization_62_layer_call_and_return_conditional_losses_7050150

inputs0
!batchnorm_readvariableop_resource:	Ü4
%batchnorm_mul_readvariableop_resource:	Ü2
#batchnorm_readvariableop_1_resource:	Ü2
#batchnorm_readvariableop_2_resource:	Ü
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Ü2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ü2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2

IdentityÂ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÜ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
 
_user_specified_nameinputs
ã
×
8__inference_batch_normalization_62_layer_call_fn_7050199

inputs
unknown:	Ü
	unknown_0:	Ü
	unknown_1:	Ü
	unknown_2:	Ü
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_62_layer_call_and_return_conditional_losses_70480262
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÜ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
 
_user_specified_nameinputs

°
E__inference_dense_53_layer_call_and_return_conditional_losses_7048586

inputs2
matmul_readvariableop_resource:
ÜÜ
identity¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ÜÜ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
MatMull
IdentityIdentityMatMul:product:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:ÿÿÿÿÿÿÿÿÿÜ: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
 
_user_specified_nameinputs
Ñ

*__inference_dense_53_layer_call_fn_7050432

inputs
unknown:
ÜÜ
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_53_layer_call_and_return_conditional_losses_70485862
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:ÿÿÿÿÿÿÿÿÿÜ: 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
 
_user_specified_nameinputs
ö,
ð
S__inference_batch_normalization_64_layer_call_and_return_conditional_losses_7048420

inputs6
'assignmovingavg_readvariableop_resource:	Ü8
)assignmovingavg_1_readvariableop_resource:	Ü4
%batchnorm_mul_readvariableop_resource:	Ü0
!batchnorm_readvariableop_resource:	Ü
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	Ü*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	Ü2
moments/StopGradient¥
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices³
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	Ü*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:Ü*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:Ü*
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
AssignMovingAvg/Cast¥
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:Ü*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:Ü2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes	
:Ü2
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
AssignMovingAvg_1/Cast«
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:Ü*
dtype02"
 AssignMovingAvg_1/ReadVariableOp¡
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:Ü2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes	
:Ü2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1k
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Ü2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ü2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2

Identityò
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÜ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
 
_user_specified_nameinputs
Ñ

*__inference_dense_50_layer_call_fn_7050390

inputs
unknown:
ÈÜ
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_50_layer_call_and_return_conditional_losses_70485232
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ: 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
ö,
ð
S__inference_batch_normalization_63_layer_call_and_return_conditional_losses_7048254

inputs6
'assignmovingavg_readvariableop_resource:	Ü8
)assignmovingavg_1_readvariableop_resource:	Ü4
%batchnorm_mul_readvariableop_resource:	Ü0
!batchnorm_readvariableop_resource:	Ü
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	Ü*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	Ü2
moments/StopGradient¥
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices³
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	Ü*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:Ü*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:Ü*
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
AssignMovingAvg/Cast¥
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:Ü*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:Ü2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes	
:Ü2
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
AssignMovingAvg_1/Cast«
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:Ü*
dtype02"
 AssignMovingAvg_1/ReadVariableOp¡
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:Ü2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes	
:Ü2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1k
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Ü2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ü2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2

Identityò
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÜ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
 
_user_specified_nameinputs
¿á

T__inference_feed_forward_sub_net_10_layer_call_and_return_conditional_losses_7049446
xM
>batch_normalization_60_assignmovingavg_readvariableop_resource:	ÈO
@batch_normalization_60_assignmovingavg_1_readvariableop_resource:	ÈK
<batch_normalization_60_batchnorm_mul_readvariableop_resource:	ÈG
8batch_normalization_60_batchnorm_readvariableop_resource:	È;
'dense_50_matmul_readvariableop_resource:
ÈÜM
>batch_normalization_61_assignmovingavg_readvariableop_resource:	ÜO
@batch_normalization_61_assignmovingavg_1_readvariableop_resource:	ÜK
<batch_normalization_61_batchnorm_mul_readvariableop_resource:	ÜG
8batch_normalization_61_batchnorm_readvariableop_resource:	Ü;
'dense_51_matmul_readvariableop_resource:
ÜÜM
>batch_normalization_62_assignmovingavg_readvariableop_resource:	ÜO
@batch_normalization_62_assignmovingavg_1_readvariableop_resource:	ÜK
<batch_normalization_62_batchnorm_mul_readvariableop_resource:	ÜG
8batch_normalization_62_batchnorm_readvariableop_resource:	Ü;
'dense_52_matmul_readvariableop_resource:
ÜÜM
>batch_normalization_63_assignmovingavg_readvariableop_resource:	ÜO
@batch_normalization_63_assignmovingavg_1_readvariableop_resource:	ÜK
<batch_normalization_63_batchnorm_mul_readvariableop_resource:	ÜG
8batch_normalization_63_batchnorm_readvariableop_resource:	Ü;
'dense_53_matmul_readvariableop_resource:
ÜÜM
>batch_normalization_64_assignmovingavg_readvariableop_resource:	ÜO
@batch_normalization_64_assignmovingavg_1_readvariableop_resource:	ÜK
<batch_normalization_64_batchnorm_mul_readvariableop_resource:	ÜG
8batch_normalization_64_batchnorm_readvariableop_resource:	Ü;
'dense_54_matmul_readvariableop_resource:
ÜÈ7
(dense_54_biasadd_readvariableop_resource:	È
identity¢&batch_normalization_60/AssignMovingAvg¢5batch_normalization_60/AssignMovingAvg/ReadVariableOp¢(batch_normalization_60/AssignMovingAvg_1¢7batch_normalization_60/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_60/batchnorm/ReadVariableOp¢3batch_normalization_60/batchnorm/mul/ReadVariableOp¢&batch_normalization_61/AssignMovingAvg¢5batch_normalization_61/AssignMovingAvg/ReadVariableOp¢(batch_normalization_61/AssignMovingAvg_1¢7batch_normalization_61/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_61/batchnorm/ReadVariableOp¢3batch_normalization_61/batchnorm/mul/ReadVariableOp¢&batch_normalization_62/AssignMovingAvg¢5batch_normalization_62/AssignMovingAvg/ReadVariableOp¢(batch_normalization_62/AssignMovingAvg_1¢7batch_normalization_62/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_62/batchnorm/ReadVariableOp¢3batch_normalization_62/batchnorm/mul/ReadVariableOp¢&batch_normalization_63/AssignMovingAvg¢5batch_normalization_63/AssignMovingAvg/ReadVariableOp¢(batch_normalization_63/AssignMovingAvg_1¢7batch_normalization_63/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_63/batchnorm/ReadVariableOp¢3batch_normalization_63/batchnorm/mul/ReadVariableOp¢&batch_normalization_64/AssignMovingAvg¢5batch_normalization_64/AssignMovingAvg/ReadVariableOp¢(batch_normalization_64/AssignMovingAvg_1¢7batch_normalization_64/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_64/batchnorm/ReadVariableOp¢3batch_normalization_64/batchnorm/mul/ReadVariableOp¢dense_50/MatMul/ReadVariableOp¢dense_51/MatMul/ReadVariableOp¢dense_52/MatMul/ReadVariableOp¢dense_53/MatMul/ReadVariableOp¢dense_54/BiasAdd/ReadVariableOp¢dense_54/MatMul/ReadVariableOp¸
5batch_normalization_60/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_60/moments/mean/reduction_indicesÐ
#batch_normalization_60/moments/meanMeanx>batch_normalization_60/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	È*
	keep_dims(2%
#batch_normalization_60/moments/meanÂ
+batch_normalization_60/moments/StopGradientStopGradient,batch_normalization_60/moments/mean:output:0*
T0*
_output_shapes
:	È2-
+batch_normalization_60/moments/StopGradientå
0batch_normalization_60/moments/SquaredDifferenceSquaredDifferencex4batch_normalization_60/moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ22
0batch_normalization_60/moments/SquaredDifferenceÀ
9batch_normalization_60/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_60/moments/variance/reduction_indices
'batch_normalization_60/moments/varianceMean4batch_normalization_60/moments/SquaredDifference:z:0Bbatch_normalization_60/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	È*
	keep_dims(2)
'batch_normalization_60/moments/varianceÆ
&batch_normalization_60/moments/SqueezeSqueeze,batch_normalization_60/moments/mean:output:0*
T0*
_output_shapes	
:È*
squeeze_dims
 2(
&batch_normalization_60/moments/SqueezeÎ
(batch_normalization_60/moments/Squeeze_1Squeeze0batch_normalization_60/moments/variance:output:0*
T0*
_output_shapes	
:È*
squeeze_dims
 2*
(batch_normalization_60/moments/Squeeze_1¡
,batch_normalization_60/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_60/AssignMovingAvg/decayÉ
+batch_normalization_60/AssignMovingAvg/CastCast5batch_normalization_60/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_60/AssignMovingAvg/Castê
5batch_normalization_60/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_60_assignmovingavg_readvariableop_resource*
_output_shapes	
:È*
dtype027
5batch_normalization_60/AssignMovingAvg/ReadVariableOpõ
*batch_normalization_60/AssignMovingAvg/subSub=batch_normalization_60/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_60/moments/Squeeze:output:0*
T0*
_output_shapes	
:È2,
*batch_normalization_60/AssignMovingAvg/subæ
*batch_normalization_60/AssignMovingAvg/mulMul.batch_normalization_60/AssignMovingAvg/sub:z:0/batch_normalization_60/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes	
:È2,
*batch_normalization_60/AssignMovingAvg/mul²
&batch_normalization_60/AssignMovingAvgAssignSubVariableOp>batch_normalization_60_assignmovingavg_readvariableop_resource.batch_normalization_60/AssignMovingAvg/mul:z:06^batch_normalization_60/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_60/AssignMovingAvg¥
.batch_normalization_60/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_60/AssignMovingAvg_1/decayÏ
-batch_normalization_60/AssignMovingAvg_1/CastCast7batch_normalization_60/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_60/AssignMovingAvg_1/Castð
7batch_normalization_60/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_60_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:È*
dtype029
7batch_normalization_60/AssignMovingAvg_1/ReadVariableOpý
,batch_normalization_60/AssignMovingAvg_1/subSub?batch_normalization_60/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_60/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:È2.
,batch_normalization_60/AssignMovingAvg_1/subî
,batch_normalization_60/AssignMovingAvg_1/mulMul0batch_normalization_60/AssignMovingAvg_1/sub:z:01batch_normalization_60/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes	
:È2.
,batch_normalization_60/AssignMovingAvg_1/mul¼
(batch_normalization_60/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_60_assignmovingavg_1_readvariableop_resource0batch_normalization_60/AssignMovingAvg_1/mul:z:08^batch_normalization_60/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_60/AssignMovingAvg_1
&batch_normalization_60/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_60/batchnorm/add/yß
$batch_normalization_60/batchnorm/addAddV21batch_normalization_60/moments/Squeeze_1:output:0/batch_normalization_60/batchnorm/add/y:output:0*
T0*
_output_shapes	
:È2&
$batch_normalization_60/batchnorm/add©
&batch_normalization_60/batchnorm/RsqrtRsqrt(batch_normalization_60/batchnorm/add:z:0*
T0*
_output_shapes	
:È2(
&batch_normalization_60/batchnorm/Rsqrtä
3batch_normalization_60/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_60_batchnorm_mul_readvariableop_resource*
_output_shapes	
:È*
dtype025
3batch_normalization_60/batchnorm/mul/ReadVariableOpâ
$batch_normalization_60/batchnorm/mulMul*batch_normalization_60/batchnorm/Rsqrt:y:0;batch_normalization_60/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:È2&
$batch_normalization_60/batchnorm/mul·
&batch_normalization_60/batchnorm/mul_1Mulx(batch_normalization_60/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2(
&batch_normalization_60/batchnorm/mul_1Ø
&batch_normalization_60/batchnorm/mul_2Mul/batch_normalization_60/moments/Squeeze:output:0(batch_normalization_60/batchnorm/mul:z:0*
T0*
_output_shapes	
:È2(
&batch_normalization_60/batchnorm/mul_2Ø
/batch_normalization_60/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_60_batchnorm_readvariableop_resource*
_output_shapes	
:È*
dtype021
/batch_normalization_60/batchnorm/ReadVariableOpÞ
$batch_normalization_60/batchnorm/subSub7batch_normalization_60/batchnorm/ReadVariableOp:value:0*batch_normalization_60/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:È2&
$batch_normalization_60/batchnorm/subâ
&batch_normalization_60/batchnorm/add_1AddV2*batch_normalization_60/batchnorm/mul_1:z:0(batch_normalization_60/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2(
&batch_normalization_60/batchnorm/add_1ª
dense_50/MatMul/ReadVariableOpReadVariableOp'dense_50_matmul_readvariableop_resource* 
_output_shapes
:
ÈÜ*
dtype02 
dense_50/MatMul/ReadVariableOp³
dense_50/MatMulMatMul*batch_normalization_60/batchnorm/add_1:z:0&dense_50/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
dense_50/MatMul¸
5batch_normalization_61/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_61/moments/mean/reduction_indicesè
#batch_normalization_61/moments/meanMeandense_50/MatMul:product:0>batch_normalization_61/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	Ü*
	keep_dims(2%
#batch_normalization_61/moments/meanÂ
+batch_normalization_61/moments/StopGradientStopGradient,batch_normalization_61/moments/mean:output:0*
T0*
_output_shapes
:	Ü2-
+batch_normalization_61/moments/StopGradientý
0batch_normalization_61/moments/SquaredDifferenceSquaredDifferencedense_50/MatMul:product:04batch_normalization_61/moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ22
0batch_normalization_61/moments/SquaredDifferenceÀ
9batch_normalization_61/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_61/moments/variance/reduction_indices
'batch_normalization_61/moments/varianceMean4batch_normalization_61/moments/SquaredDifference:z:0Bbatch_normalization_61/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	Ü*
	keep_dims(2)
'batch_normalization_61/moments/varianceÆ
&batch_normalization_61/moments/SqueezeSqueeze,batch_normalization_61/moments/mean:output:0*
T0*
_output_shapes	
:Ü*
squeeze_dims
 2(
&batch_normalization_61/moments/SqueezeÎ
(batch_normalization_61/moments/Squeeze_1Squeeze0batch_normalization_61/moments/variance:output:0*
T0*
_output_shapes	
:Ü*
squeeze_dims
 2*
(batch_normalization_61/moments/Squeeze_1¡
,batch_normalization_61/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_61/AssignMovingAvg/decayÉ
+batch_normalization_61/AssignMovingAvg/CastCast5batch_normalization_61/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_61/AssignMovingAvg/Castê
5batch_normalization_61/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_61_assignmovingavg_readvariableop_resource*
_output_shapes	
:Ü*
dtype027
5batch_normalization_61/AssignMovingAvg/ReadVariableOpõ
*batch_normalization_61/AssignMovingAvg/subSub=batch_normalization_61/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_61/moments/Squeeze:output:0*
T0*
_output_shapes	
:Ü2,
*batch_normalization_61/AssignMovingAvg/subæ
*batch_normalization_61/AssignMovingAvg/mulMul.batch_normalization_61/AssignMovingAvg/sub:z:0/batch_normalization_61/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes	
:Ü2,
*batch_normalization_61/AssignMovingAvg/mul²
&batch_normalization_61/AssignMovingAvgAssignSubVariableOp>batch_normalization_61_assignmovingavg_readvariableop_resource.batch_normalization_61/AssignMovingAvg/mul:z:06^batch_normalization_61/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_61/AssignMovingAvg¥
.batch_normalization_61/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_61/AssignMovingAvg_1/decayÏ
-batch_normalization_61/AssignMovingAvg_1/CastCast7batch_normalization_61/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_61/AssignMovingAvg_1/Castð
7batch_normalization_61/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_61_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:Ü*
dtype029
7batch_normalization_61/AssignMovingAvg_1/ReadVariableOpý
,batch_normalization_61/AssignMovingAvg_1/subSub?batch_normalization_61/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_61/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:Ü2.
,batch_normalization_61/AssignMovingAvg_1/subî
,batch_normalization_61/AssignMovingAvg_1/mulMul0batch_normalization_61/AssignMovingAvg_1/sub:z:01batch_normalization_61/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes	
:Ü2.
,batch_normalization_61/AssignMovingAvg_1/mul¼
(batch_normalization_61/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_61_assignmovingavg_1_readvariableop_resource0batch_normalization_61/AssignMovingAvg_1/mul:z:08^batch_normalization_61/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_61/AssignMovingAvg_1
&batch_normalization_61/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_61/batchnorm/add/yß
$batch_normalization_61/batchnorm/addAddV21batch_normalization_61/moments/Squeeze_1:output:0/batch_normalization_61/batchnorm/add/y:output:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_61/batchnorm/add©
&batch_normalization_61/batchnorm/RsqrtRsqrt(batch_normalization_61/batchnorm/add:z:0*
T0*
_output_shapes	
:Ü2(
&batch_normalization_61/batchnorm/Rsqrtä
3batch_normalization_61/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_61_batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ü*
dtype025
3batch_normalization_61/batchnorm/mul/ReadVariableOpâ
$batch_normalization_61/batchnorm/mulMul*batch_normalization_61/batchnorm/Rsqrt:y:0;batch_normalization_61/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_61/batchnorm/mulÏ
&batch_normalization_61/batchnorm/mul_1Muldense_50/MatMul:product:0(batch_normalization_61/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2(
&batch_normalization_61/batchnorm/mul_1Ø
&batch_normalization_61/batchnorm/mul_2Mul/batch_normalization_61/moments/Squeeze:output:0(batch_normalization_61/batchnorm/mul:z:0*
T0*
_output_shapes	
:Ü2(
&batch_normalization_61/batchnorm/mul_2Ø
/batch_normalization_61/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_61_batchnorm_readvariableop_resource*
_output_shapes	
:Ü*
dtype021
/batch_normalization_61/batchnorm/ReadVariableOpÞ
$batch_normalization_61/batchnorm/subSub7batch_normalization_61/batchnorm/ReadVariableOp:value:0*batch_normalization_61/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_61/batchnorm/subâ
&batch_normalization_61/batchnorm/add_1AddV2*batch_normalization_61/batchnorm/mul_1:z:0(batch_normalization_61/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2(
&batch_normalization_61/batchnorm/add_1s
ReluRelu*batch_normalization_61/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
Reluª
dense_51/MatMul/ReadVariableOpReadVariableOp'dense_51_matmul_readvariableop_resource* 
_output_shapes
:
ÜÜ*
dtype02 
dense_51/MatMul/ReadVariableOp
dense_51/MatMulMatMulRelu:activations:0&dense_51/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
dense_51/MatMul¸
5batch_normalization_62/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_62/moments/mean/reduction_indicesè
#batch_normalization_62/moments/meanMeandense_51/MatMul:product:0>batch_normalization_62/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	Ü*
	keep_dims(2%
#batch_normalization_62/moments/meanÂ
+batch_normalization_62/moments/StopGradientStopGradient,batch_normalization_62/moments/mean:output:0*
T0*
_output_shapes
:	Ü2-
+batch_normalization_62/moments/StopGradientý
0batch_normalization_62/moments/SquaredDifferenceSquaredDifferencedense_51/MatMul:product:04batch_normalization_62/moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ22
0batch_normalization_62/moments/SquaredDifferenceÀ
9batch_normalization_62/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_62/moments/variance/reduction_indices
'batch_normalization_62/moments/varianceMean4batch_normalization_62/moments/SquaredDifference:z:0Bbatch_normalization_62/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	Ü*
	keep_dims(2)
'batch_normalization_62/moments/varianceÆ
&batch_normalization_62/moments/SqueezeSqueeze,batch_normalization_62/moments/mean:output:0*
T0*
_output_shapes	
:Ü*
squeeze_dims
 2(
&batch_normalization_62/moments/SqueezeÎ
(batch_normalization_62/moments/Squeeze_1Squeeze0batch_normalization_62/moments/variance:output:0*
T0*
_output_shapes	
:Ü*
squeeze_dims
 2*
(batch_normalization_62/moments/Squeeze_1¡
,batch_normalization_62/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_62/AssignMovingAvg/decayÉ
+batch_normalization_62/AssignMovingAvg/CastCast5batch_normalization_62/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_62/AssignMovingAvg/Castê
5batch_normalization_62/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_62_assignmovingavg_readvariableop_resource*
_output_shapes	
:Ü*
dtype027
5batch_normalization_62/AssignMovingAvg/ReadVariableOpõ
*batch_normalization_62/AssignMovingAvg/subSub=batch_normalization_62/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_62/moments/Squeeze:output:0*
T0*
_output_shapes	
:Ü2,
*batch_normalization_62/AssignMovingAvg/subæ
*batch_normalization_62/AssignMovingAvg/mulMul.batch_normalization_62/AssignMovingAvg/sub:z:0/batch_normalization_62/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes	
:Ü2,
*batch_normalization_62/AssignMovingAvg/mul²
&batch_normalization_62/AssignMovingAvgAssignSubVariableOp>batch_normalization_62_assignmovingavg_readvariableop_resource.batch_normalization_62/AssignMovingAvg/mul:z:06^batch_normalization_62/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_62/AssignMovingAvg¥
.batch_normalization_62/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_62/AssignMovingAvg_1/decayÏ
-batch_normalization_62/AssignMovingAvg_1/CastCast7batch_normalization_62/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_62/AssignMovingAvg_1/Castð
7batch_normalization_62/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_62_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:Ü*
dtype029
7batch_normalization_62/AssignMovingAvg_1/ReadVariableOpý
,batch_normalization_62/AssignMovingAvg_1/subSub?batch_normalization_62/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_62/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:Ü2.
,batch_normalization_62/AssignMovingAvg_1/subî
,batch_normalization_62/AssignMovingAvg_1/mulMul0batch_normalization_62/AssignMovingAvg_1/sub:z:01batch_normalization_62/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes	
:Ü2.
,batch_normalization_62/AssignMovingAvg_1/mul¼
(batch_normalization_62/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_62_assignmovingavg_1_readvariableop_resource0batch_normalization_62/AssignMovingAvg_1/mul:z:08^batch_normalization_62/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_62/AssignMovingAvg_1
&batch_normalization_62/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_62/batchnorm/add/yß
$batch_normalization_62/batchnorm/addAddV21batch_normalization_62/moments/Squeeze_1:output:0/batch_normalization_62/batchnorm/add/y:output:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_62/batchnorm/add©
&batch_normalization_62/batchnorm/RsqrtRsqrt(batch_normalization_62/batchnorm/add:z:0*
T0*
_output_shapes	
:Ü2(
&batch_normalization_62/batchnorm/Rsqrtä
3batch_normalization_62/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_62_batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ü*
dtype025
3batch_normalization_62/batchnorm/mul/ReadVariableOpâ
$batch_normalization_62/batchnorm/mulMul*batch_normalization_62/batchnorm/Rsqrt:y:0;batch_normalization_62/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_62/batchnorm/mulÏ
&batch_normalization_62/batchnorm/mul_1Muldense_51/MatMul:product:0(batch_normalization_62/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2(
&batch_normalization_62/batchnorm/mul_1Ø
&batch_normalization_62/batchnorm/mul_2Mul/batch_normalization_62/moments/Squeeze:output:0(batch_normalization_62/batchnorm/mul:z:0*
T0*
_output_shapes	
:Ü2(
&batch_normalization_62/batchnorm/mul_2Ø
/batch_normalization_62/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_62_batchnorm_readvariableop_resource*
_output_shapes	
:Ü*
dtype021
/batch_normalization_62/batchnorm/ReadVariableOpÞ
$batch_normalization_62/batchnorm/subSub7batch_normalization_62/batchnorm/ReadVariableOp:value:0*batch_normalization_62/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_62/batchnorm/subâ
&batch_normalization_62/batchnorm/add_1AddV2*batch_normalization_62/batchnorm/mul_1:z:0(batch_normalization_62/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2(
&batch_normalization_62/batchnorm/add_1w
Relu_1Relu*batch_normalization_62/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
Relu_1ª
dense_52/MatMul/ReadVariableOpReadVariableOp'dense_52_matmul_readvariableop_resource* 
_output_shapes
:
ÜÜ*
dtype02 
dense_52/MatMul/ReadVariableOp
dense_52/MatMulMatMulRelu_1:activations:0&dense_52/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
dense_52/MatMul¸
5batch_normalization_63/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_63/moments/mean/reduction_indicesè
#batch_normalization_63/moments/meanMeandense_52/MatMul:product:0>batch_normalization_63/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	Ü*
	keep_dims(2%
#batch_normalization_63/moments/meanÂ
+batch_normalization_63/moments/StopGradientStopGradient,batch_normalization_63/moments/mean:output:0*
T0*
_output_shapes
:	Ü2-
+batch_normalization_63/moments/StopGradientý
0batch_normalization_63/moments/SquaredDifferenceSquaredDifferencedense_52/MatMul:product:04batch_normalization_63/moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ22
0batch_normalization_63/moments/SquaredDifferenceÀ
9batch_normalization_63/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_63/moments/variance/reduction_indices
'batch_normalization_63/moments/varianceMean4batch_normalization_63/moments/SquaredDifference:z:0Bbatch_normalization_63/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	Ü*
	keep_dims(2)
'batch_normalization_63/moments/varianceÆ
&batch_normalization_63/moments/SqueezeSqueeze,batch_normalization_63/moments/mean:output:0*
T0*
_output_shapes	
:Ü*
squeeze_dims
 2(
&batch_normalization_63/moments/SqueezeÎ
(batch_normalization_63/moments/Squeeze_1Squeeze0batch_normalization_63/moments/variance:output:0*
T0*
_output_shapes	
:Ü*
squeeze_dims
 2*
(batch_normalization_63/moments/Squeeze_1¡
,batch_normalization_63/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_63/AssignMovingAvg/decayÉ
+batch_normalization_63/AssignMovingAvg/CastCast5batch_normalization_63/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_63/AssignMovingAvg/Castê
5batch_normalization_63/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_63_assignmovingavg_readvariableop_resource*
_output_shapes	
:Ü*
dtype027
5batch_normalization_63/AssignMovingAvg/ReadVariableOpõ
*batch_normalization_63/AssignMovingAvg/subSub=batch_normalization_63/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_63/moments/Squeeze:output:0*
T0*
_output_shapes	
:Ü2,
*batch_normalization_63/AssignMovingAvg/subæ
*batch_normalization_63/AssignMovingAvg/mulMul.batch_normalization_63/AssignMovingAvg/sub:z:0/batch_normalization_63/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes	
:Ü2,
*batch_normalization_63/AssignMovingAvg/mul²
&batch_normalization_63/AssignMovingAvgAssignSubVariableOp>batch_normalization_63_assignmovingavg_readvariableop_resource.batch_normalization_63/AssignMovingAvg/mul:z:06^batch_normalization_63/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_63/AssignMovingAvg¥
.batch_normalization_63/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_63/AssignMovingAvg_1/decayÏ
-batch_normalization_63/AssignMovingAvg_1/CastCast7batch_normalization_63/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_63/AssignMovingAvg_1/Castð
7batch_normalization_63/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_63_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:Ü*
dtype029
7batch_normalization_63/AssignMovingAvg_1/ReadVariableOpý
,batch_normalization_63/AssignMovingAvg_1/subSub?batch_normalization_63/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_63/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:Ü2.
,batch_normalization_63/AssignMovingAvg_1/subî
,batch_normalization_63/AssignMovingAvg_1/mulMul0batch_normalization_63/AssignMovingAvg_1/sub:z:01batch_normalization_63/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes	
:Ü2.
,batch_normalization_63/AssignMovingAvg_1/mul¼
(batch_normalization_63/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_63_assignmovingavg_1_readvariableop_resource0batch_normalization_63/AssignMovingAvg_1/mul:z:08^batch_normalization_63/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_63/AssignMovingAvg_1
&batch_normalization_63/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_63/batchnorm/add/yß
$batch_normalization_63/batchnorm/addAddV21batch_normalization_63/moments/Squeeze_1:output:0/batch_normalization_63/batchnorm/add/y:output:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_63/batchnorm/add©
&batch_normalization_63/batchnorm/RsqrtRsqrt(batch_normalization_63/batchnorm/add:z:0*
T0*
_output_shapes	
:Ü2(
&batch_normalization_63/batchnorm/Rsqrtä
3batch_normalization_63/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_63_batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ü*
dtype025
3batch_normalization_63/batchnorm/mul/ReadVariableOpâ
$batch_normalization_63/batchnorm/mulMul*batch_normalization_63/batchnorm/Rsqrt:y:0;batch_normalization_63/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_63/batchnorm/mulÏ
&batch_normalization_63/batchnorm/mul_1Muldense_52/MatMul:product:0(batch_normalization_63/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2(
&batch_normalization_63/batchnorm/mul_1Ø
&batch_normalization_63/batchnorm/mul_2Mul/batch_normalization_63/moments/Squeeze:output:0(batch_normalization_63/batchnorm/mul:z:0*
T0*
_output_shapes	
:Ü2(
&batch_normalization_63/batchnorm/mul_2Ø
/batch_normalization_63/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_63_batchnorm_readvariableop_resource*
_output_shapes	
:Ü*
dtype021
/batch_normalization_63/batchnorm/ReadVariableOpÞ
$batch_normalization_63/batchnorm/subSub7batch_normalization_63/batchnorm/ReadVariableOp:value:0*batch_normalization_63/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_63/batchnorm/subâ
&batch_normalization_63/batchnorm/add_1AddV2*batch_normalization_63/batchnorm/mul_1:z:0(batch_normalization_63/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2(
&batch_normalization_63/batchnorm/add_1w
Relu_2Relu*batch_normalization_63/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
Relu_2ª
dense_53/MatMul/ReadVariableOpReadVariableOp'dense_53_matmul_readvariableop_resource* 
_output_shapes
:
ÜÜ*
dtype02 
dense_53/MatMul/ReadVariableOp
dense_53/MatMulMatMulRelu_2:activations:0&dense_53/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
dense_53/MatMul¸
5batch_normalization_64/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_64/moments/mean/reduction_indicesè
#batch_normalization_64/moments/meanMeandense_53/MatMul:product:0>batch_normalization_64/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	Ü*
	keep_dims(2%
#batch_normalization_64/moments/meanÂ
+batch_normalization_64/moments/StopGradientStopGradient,batch_normalization_64/moments/mean:output:0*
T0*
_output_shapes
:	Ü2-
+batch_normalization_64/moments/StopGradientý
0batch_normalization_64/moments/SquaredDifferenceSquaredDifferencedense_53/MatMul:product:04batch_normalization_64/moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ22
0batch_normalization_64/moments/SquaredDifferenceÀ
9batch_normalization_64/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_64/moments/variance/reduction_indices
'batch_normalization_64/moments/varianceMean4batch_normalization_64/moments/SquaredDifference:z:0Bbatch_normalization_64/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	Ü*
	keep_dims(2)
'batch_normalization_64/moments/varianceÆ
&batch_normalization_64/moments/SqueezeSqueeze,batch_normalization_64/moments/mean:output:0*
T0*
_output_shapes	
:Ü*
squeeze_dims
 2(
&batch_normalization_64/moments/SqueezeÎ
(batch_normalization_64/moments/Squeeze_1Squeeze0batch_normalization_64/moments/variance:output:0*
T0*
_output_shapes	
:Ü*
squeeze_dims
 2*
(batch_normalization_64/moments/Squeeze_1¡
,batch_normalization_64/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_64/AssignMovingAvg/decayÉ
+batch_normalization_64/AssignMovingAvg/CastCast5batch_normalization_64/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_64/AssignMovingAvg/Castê
5batch_normalization_64/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_64_assignmovingavg_readvariableop_resource*
_output_shapes	
:Ü*
dtype027
5batch_normalization_64/AssignMovingAvg/ReadVariableOpõ
*batch_normalization_64/AssignMovingAvg/subSub=batch_normalization_64/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_64/moments/Squeeze:output:0*
T0*
_output_shapes	
:Ü2,
*batch_normalization_64/AssignMovingAvg/subæ
*batch_normalization_64/AssignMovingAvg/mulMul.batch_normalization_64/AssignMovingAvg/sub:z:0/batch_normalization_64/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes	
:Ü2,
*batch_normalization_64/AssignMovingAvg/mul²
&batch_normalization_64/AssignMovingAvgAssignSubVariableOp>batch_normalization_64_assignmovingavg_readvariableop_resource.batch_normalization_64/AssignMovingAvg/mul:z:06^batch_normalization_64/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_64/AssignMovingAvg¥
.batch_normalization_64/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_64/AssignMovingAvg_1/decayÏ
-batch_normalization_64/AssignMovingAvg_1/CastCast7batch_normalization_64/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_64/AssignMovingAvg_1/Castð
7batch_normalization_64/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_64_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:Ü*
dtype029
7batch_normalization_64/AssignMovingAvg_1/ReadVariableOpý
,batch_normalization_64/AssignMovingAvg_1/subSub?batch_normalization_64/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_64/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:Ü2.
,batch_normalization_64/AssignMovingAvg_1/subî
,batch_normalization_64/AssignMovingAvg_1/mulMul0batch_normalization_64/AssignMovingAvg_1/sub:z:01batch_normalization_64/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes	
:Ü2.
,batch_normalization_64/AssignMovingAvg_1/mul¼
(batch_normalization_64/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_64_assignmovingavg_1_readvariableop_resource0batch_normalization_64/AssignMovingAvg_1/mul:z:08^batch_normalization_64/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_64/AssignMovingAvg_1
&batch_normalization_64/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_64/batchnorm/add/yß
$batch_normalization_64/batchnorm/addAddV21batch_normalization_64/moments/Squeeze_1:output:0/batch_normalization_64/batchnorm/add/y:output:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_64/batchnorm/add©
&batch_normalization_64/batchnorm/RsqrtRsqrt(batch_normalization_64/batchnorm/add:z:0*
T0*
_output_shapes	
:Ü2(
&batch_normalization_64/batchnorm/Rsqrtä
3batch_normalization_64/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_64_batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ü*
dtype025
3batch_normalization_64/batchnorm/mul/ReadVariableOpâ
$batch_normalization_64/batchnorm/mulMul*batch_normalization_64/batchnorm/Rsqrt:y:0;batch_normalization_64/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_64/batchnorm/mulÏ
&batch_normalization_64/batchnorm/mul_1Muldense_53/MatMul:product:0(batch_normalization_64/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2(
&batch_normalization_64/batchnorm/mul_1Ø
&batch_normalization_64/batchnorm/mul_2Mul/batch_normalization_64/moments/Squeeze:output:0(batch_normalization_64/batchnorm/mul:z:0*
T0*
_output_shapes	
:Ü2(
&batch_normalization_64/batchnorm/mul_2Ø
/batch_normalization_64/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_64_batchnorm_readvariableop_resource*
_output_shapes	
:Ü*
dtype021
/batch_normalization_64/batchnorm/ReadVariableOpÞ
$batch_normalization_64/batchnorm/subSub7batch_normalization_64/batchnorm/ReadVariableOp:value:0*batch_normalization_64/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_64/batchnorm/subâ
&batch_normalization_64/batchnorm/add_1AddV2*batch_normalization_64/batchnorm/mul_1:z:0(batch_normalization_64/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2(
&batch_normalization_64/batchnorm/add_1w
Relu_3Relu*batch_normalization_64/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
Relu_3ª
dense_54/MatMul/ReadVariableOpReadVariableOp'dense_54_matmul_readvariableop_resource* 
_output_shapes
:
ÜÈ*
dtype02 
dense_54/MatMul/ReadVariableOp
dense_54/MatMulMatMulRelu_3:activations:0&dense_54/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
dense_54/MatMul¨
dense_54/BiasAdd/ReadVariableOpReadVariableOp(dense_54_biasadd_readvariableop_resource*
_output_shapes	
:È*
dtype02!
dense_54/BiasAdd/ReadVariableOp¦
dense_54/BiasAddBiasAdddense_54/MatMul:product:0'dense_54/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
dense_54/BiasAddu
IdentityIdentitydense_54/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identityû
NoOpNoOp'^batch_normalization_60/AssignMovingAvg6^batch_normalization_60/AssignMovingAvg/ReadVariableOp)^batch_normalization_60/AssignMovingAvg_18^batch_normalization_60/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_60/batchnorm/ReadVariableOp4^batch_normalization_60/batchnorm/mul/ReadVariableOp'^batch_normalization_61/AssignMovingAvg6^batch_normalization_61/AssignMovingAvg/ReadVariableOp)^batch_normalization_61/AssignMovingAvg_18^batch_normalization_61/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_61/batchnorm/ReadVariableOp4^batch_normalization_61/batchnorm/mul/ReadVariableOp'^batch_normalization_62/AssignMovingAvg6^batch_normalization_62/AssignMovingAvg/ReadVariableOp)^batch_normalization_62/AssignMovingAvg_18^batch_normalization_62/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_62/batchnorm/ReadVariableOp4^batch_normalization_62/batchnorm/mul/ReadVariableOp'^batch_normalization_63/AssignMovingAvg6^batch_normalization_63/AssignMovingAvg/ReadVariableOp)^batch_normalization_63/AssignMovingAvg_18^batch_normalization_63/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_63/batchnorm/ReadVariableOp4^batch_normalization_63/batchnorm/mul/ReadVariableOp'^batch_normalization_64/AssignMovingAvg6^batch_normalization_64/AssignMovingAvg/ReadVariableOp)^batch_normalization_64/AssignMovingAvg_18^batch_normalization_64/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_64/batchnorm/ReadVariableOp4^batch_normalization_64/batchnorm/mul/ReadVariableOp^dense_50/MatMul/ReadVariableOp^dense_51/MatMul/ReadVariableOp^dense_52/MatMul/ReadVariableOp^dense_53/MatMul/ReadVariableOp ^dense_54/BiasAdd/ReadVariableOp^dense_54/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:ÿÿÿÿÿÿÿÿÿÈ: : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_60/AssignMovingAvg&batch_normalization_60/AssignMovingAvg2n
5batch_normalization_60/AssignMovingAvg/ReadVariableOp5batch_normalization_60/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_60/AssignMovingAvg_1(batch_normalization_60/AssignMovingAvg_12r
7batch_normalization_60/AssignMovingAvg_1/ReadVariableOp7batch_normalization_60/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_60/batchnorm/ReadVariableOp/batch_normalization_60/batchnorm/ReadVariableOp2j
3batch_normalization_60/batchnorm/mul/ReadVariableOp3batch_normalization_60/batchnorm/mul/ReadVariableOp2P
&batch_normalization_61/AssignMovingAvg&batch_normalization_61/AssignMovingAvg2n
5batch_normalization_61/AssignMovingAvg/ReadVariableOp5batch_normalization_61/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_61/AssignMovingAvg_1(batch_normalization_61/AssignMovingAvg_12r
7batch_normalization_61/AssignMovingAvg_1/ReadVariableOp7batch_normalization_61/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_61/batchnorm/ReadVariableOp/batch_normalization_61/batchnorm/ReadVariableOp2j
3batch_normalization_61/batchnorm/mul/ReadVariableOp3batch_normalization_61/batchnorm/mul/ReadVariableOp2P
&batch_normalization_62/AssignMovingAvg&batch_normalization_62/AssignMovingAvg2n
5batch_normalization_62/AssignMovingAvg/ReadVariableOp5batch_normalization_62/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_62/AssignMovingAvg_1(batch_normalization_62/AssignMovingAvg_12r
7batch_normalization_62/AssignMovingAvg_1/ReadVariableOp7batch_normalization_62/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_62/batchnorm/ReadVariableOp/batch_normalization_62/batchnorm/ReadVariableOp2j
3batch_normalization_62/batchnorm/mul/ReadVariableOp3batch_normalization_62/batchnorm/mul/ReadVariableOp2P
&batch_normalization_63/AssignMovingAvg&batch_normalization_63/AssignMovingAvg2n
5batch_normalization_63/AssignMovingAvg/ReadVariableOp5batch_normalization_63/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_63/AssignMovingAvg_1(batch_normalization_63/AssignMovingAvg_12r
7batch_normalization_63/AssignMovingAvg_1/ReadVariableOp7batch_normalization_63/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_63/batchnorm/ReadVariableOp/batch_normalization_63/batchnorm/ReadVariableOp2j
3batch_normalization_63/batchnorm/mul/ReadVariableOp3batch_normalization_63/batchnorm/mul/ReadVariableOp2P
&batch_normalization_64/AssignMovingAvg&batch_normalization_64/AssignMovingAvg2n
5batch_normalization_64/AssignMovingAvg/ReadVariableOp5batch_normalization_64/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_64/AssignMovingAvg_1(batch_normalization_64/AssignMovingAvg_12r
7batch_normalization_64/AssignMovingAvg_1/ReadVariableOp7batch_normalization_64/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_64/batchnorm/ReadVariableOp/batch_normalization_64/batchnorm/ReadVariableOp2j
3batch_normalization_64/batchnorm/mul/ReadVariableOp3batch_normalization_64/batchnorm/mul/ReadVariableOp2@
dense_50/MatMul/ReadVariableOpdense_50/MatMul/ReadVariableOp2@
dense_51/MatMul/ReadVariableOpdense_51/MatMul/ReadVariableOp2@
dense_52/MatMul/ReadVariableOpdense_52/MatMul/ReadVariableOp2@
dense_53/MatMul/ReadVariableOpdense_53/MatMul/ReadVariableOp2B
dense_54/BiasAdd/ReadVariableOpdense_54/BiasAdd/ReadVariableOp2@
dense_54/MatMul/ReadVariableOpdense_54/MatMul/ReadVariableOp:K G
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ

_user_specified_namex
ö,
ð
S__inference_batch_normalization_60_layer_call_and_return_conditional_losses_7047756

inputs6
'assignmovingavg_readvariableop_resource:	È8
)assignmovingavg_1_readvariableop_resource:	È4
%batchnorm_mul_readvariableop_resource:	È0
!batchnorm_readvariableop_resource:	È
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	È*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	È2
moments/StopGradient¥
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices³
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	È*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:È*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:È*
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
AssignMovingAvg/Cast¥
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:È*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:È2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes	
:È2
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
AssignMovingAvg_1/Cast«
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:È*
dtype02"
 AssignMovingAvg_1/ReadVariableOp¡
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:È2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes	
:È2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1k
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:È2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:È2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:È*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:È2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:È2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:È*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:È2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identityò
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs

°
E__inference_dense_51_layer_call_and_return_conditional_losses_7048544

inputs2
matmul_readvariableop_resource:
ÜÜ
identity¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ÜÜ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
MatMull
IdentityIdentityMatMul:product:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:ÿÿÿÿÿÿÿÿÿÜ: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
 
_user_specified_nameinputs

¶
S__inference_batch_normalization_61_layer_call_and_return_conditional_losses_7047860

inputs0
!batchnorm_readvariableop_resource:	Ü4
%batchnorm_mul_readvariableop_resource:	Ü2
#batchnorm_readvariableop_1_resource:	Ü2
#batchnorm_readvariableop_2_resource:	Ü
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Ü2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ü2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2

IdentityÂ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÜ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
 
_user_specified_nameinputs
á
×
8__inference_batch_normalization_60_layer_call_fn_7050048

inputs
unknown:	È
	unknown_0:	È
	unknown_1:	È
	unknown_2:	È
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_60_layer_call_and_return_conditional_losses_70477562
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
²O
Þ
 __inference__traced_save_7050552
file_prefixe
asavev2_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_60_gamma_read_readvariableopd
`savev2_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_60_beta_read_readvariableope
asavev2_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_61_gamma_read_readvariableopd
`savev2_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_61_beta_read_readvariableope
asavev2_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_62_gamma_read_readvariableopd
`savev2_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_62_beta_read_readvariableope
asavev2_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_63_gamma_read_readvariableopd
`savev2_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_63_beta_read_readvariableope
asavev2_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_64_gamma_read_readvariableopd
`savev2_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_64_beta_read_readvariableopX
Tsavev2_nonshared_model_1_feed_forward_sub_net_10_dense_50_kernel_read_readvariableopX
Tsavev2_nonshared_model_1_feed_forward_sub_net_10_dense_51_kernel_read_readvariableopX
Tsavev2_nonshared_model_1_feed_forward_sub_net_10_dense_52_kernel_read_readvariableopX
Tsavev2_nonshared_model_1_feed_forward_sub_net_10_dense_53_kernel_read_readvariableopX
Tsavev2_nonshared_model_1_feed_forward_sub_net_10_dense_54_kernel_read_readvariableopV
Rsavev2_nonshared_model_1_feed_forward_sub_net_10_dense_54_bias_read_readvariableopk
gsavev2_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_60_moving_mean_read_readvariableopo
ksavev2_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_60_moving_variance_read_readvariableopk
gsavev2_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_61_moving_mean_read_readvariableopo
ksavev2_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_61_moving_variance_read_readvariableopk
gsavev2_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_62_moving_mean_read_readvariableopo
ksavev2_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_62_moving_variance_read_readvariableopk
gsavev2_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_63_moving_mean_read_readvariableopo
ksavev2_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_63_moving_variance_read_readvariableopk
gsavev2_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_64_moving_mean_read_readvariableopo
ksavev2_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_64_moving_variance_read_readvariableop
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
ShardedFilenameá

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ó	
valueé	Bæ	B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names¾
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesÝ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0asavev2_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_60_gamma_read_readvariableop`savev2_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_60_beta_read_readvariableopasavev2_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_61_gamma_read_readvariableop`savev2_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_61_beta_read_readvariableopasavev2_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_62_gamma_read_readvariableop`savev2_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_62_beta_read_readvariableopasavev2_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_63_gamma_read_readvariableop`savev2_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_63_beta_read_readvariableopasavev2_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_64_gamma_read_readvariableop`savev2_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_64_beta_read_readvariableopTsavev2_nonshared_model_1_feed_forward_sub_net_10_dense_50_kernel_read_readvariableopTsavev2_nonshared_model_1_feed_forward_sub_net_10_dense_51_kernel_read_readvariableopTsavev2_nonshared_model_1_feed_forward_sub_net_10_dense_52_kernel_read_readvariableopTsavev2_nonshared_model_1_feed_forward_sub_net_10_dense_53_kernel_read_readvariableopTsavev2_nonshared_model_1_feed_forward_sub_net_10_dense_54_kernel_read_readvariableopRsavev2_nonshared_model_1_feed_forward_sub_net_10_dense_54_bias_read_readvariableopgsavev2_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_60_moving_mean_read_readvariableopksavev2_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_60_moving_variance_read_readvariableopgsavev2_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_61_moving_mean_read_readvariableopksavev2_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_61_moving_variance_read_readvariableopgsavev2_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_62_moving_mean_read_readvariableopksavev2_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_62_moving_variance_read_readvariableopgsavev2_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_63_moving_mean_read_readvariableopksavev2_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_63_moving_variance_read_readvariableopgsavev2_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_64_moving_mean_read_readvariableopksavev2_nonshared_model_1_feed_forward_sub_net_10_batch_normalization_64_moving_variance_read_readvariableopsavev2_const"/device:CPU:0*
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

identity_1Identity_1:output:0*è
_input_shapesÖ
Ó: :È:È:Ü:Ü:Ü:Ü:Ü:Ü:Ü:Ü:
ÈÜ:
ÜÜ:
ÜÜ:
ÜÜ:
ÜÈ:È:È:È:Ü:Ü:Ü:Ü:Ü:Ü:Ü:Ü: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:!

_output_shapes	
:È:!

_output_shapes	
:È:!

_output_shapes	
:Ü:!

_output_shapes	
:Ü:!

_output_shapes	
:Ü:!

_output_shapes	
:Ü:!

_output_shapes	
:Ü:!

_output_shapes	
:Ü:!	

_output_shapes	
:Ü:!


_output_shapes	
:Ü:&"
 
_output_shapes
:
ÈÜ:&"
 
_output_shapes
:
ÜÜ:&"
 
_output_shapes
:
ÜÜ:&"
 
_output_shapes
:
ÜÜ:&"
 
_output_shapes
:
ÜÈ:!

_output_shapes	
:È:!

_output_shapes	
:È:!

_output_shapes	
:È:!

_output_shapes	
:Ü:!

_output_shapes	
:Ü:!

_output_shapes	
:Ü:!

_output_shapes	
:Ü:!

_output_shapes	
:Ü:!

_output_shapes	
:Ü:!

_output_shapes	
:Ü:!

_output_shapes	
:Ü:

_output_shapes
: 

¶
S__inference_batch_normalization_64_layer_call_and_return_conditional_losses_7048358

inputs0
!batchnorm_readvariableop_resource:	Ü4
%batchnorm_mul_readvariableop_resource:	Ü2
#batchnorm_readvariableop_1_resource:	Ü2
#batchnorm_readvariableop_2_resource:	Ü
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Ü2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ü2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2

IdentityÂ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÜ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
 
_user_specified_nameinputs

·
9__inference_feed_forward_sub_net_10_layer_call_fn_7049852
x
unknown:	È
	unknown_0:	È
	unknown_1:	È
	unknown_2:	È
	unknown_3:
ÈÜ
	unknown_4:	Ü
	unknown_5:	Ü
	unknown_6:	Ü
	unknown_7:	Ü
	unknown_8:
ÜÜ
	unknown_9:	Ü

unknown_10:	Ü

unknown_11:	Ü

unknown_12:	Ü

unknown_13:
ÜÜ

unknown_14:	Ü

unknown_15:	Ü

unknown_16:	Ü

unknown_17:	Ü

unknown_18:
ÜÜ

unknown_19:	Ü

unknown_20:	Ü

unknown_21:	Ü

unknown_22:	Ü

unknown_23:
ÜÈ

unknown_24:	È
identity¢StatefulPartitionedCallÇ
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
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_feed_forward_sub_net_10_layer_call_and_return_conditional_losses_70486172
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:ÿÿÿÿÿÿÿÿÿÈ: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ

_user_specified_namex
ö,
ð
S__inference_batch_normalization_62_layer_call_and_return_conditional_losses_7048088

inputs6
'assignmovingavg_readvariableop_resource:	Ü8
)assignmovingavg_1_readvariableop_resource:	Ü4
%batchnorm_mul_readvariableop_resource:	Ü0
!batchnorm_readvariableop_resource:	Ü
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	Ü*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	Ü2
moments/StopGradient¥
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices³
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	Ü*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:Ü*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:Ü*
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
AssignMovingAvg/Cast¥
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:Ü*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:Ü2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes	
:Ü2
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
AssignMovingAvg_1/Cast«
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:Ü*
dtype02"
 AssignMovingAvg_1/ReadVariableOp¡
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:Ü2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes	
:Ü2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1k
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Ü2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ü2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2

Identityò
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÜ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
 
_user_specified_nameinputs

¶
S__inference_batch_normalization_64_layer_call_and_return_conditional_losses_7050314

inputs0
!batchnorm_readvariableop_resource:	Ü4
%batchnorm_mul_readvariableop_resource:	Ü2
#batchnorm_readvariableop_1_resource:	Ü2
#batchnorm_readvariableop_2_resource:	Ü
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Ü2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ü2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2

IdentityÂ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÜ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
 
_user_specified_nameinputs

°
E__inference_dense_50_layer_call_and_return_conditional_losses_7050383

inputs2
matmul_readvariableop_resource:
ÈÜ
identity¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ÈÜ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
MatMull
IdentityIdentityMatMul:product:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
ã
×
8__inference_batch_normalization_60_layer_call_fn_7050035

inputs
unknown:	È
	unknown_0:	È
	unknown_1:	È
	unknown_2:	È
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_60_layer_call_and_return_conditional_losses_70476942
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
á
×
8__inference_batch_normalization_63_layer_call_fn_7050294

inputs
unknown:	Ü
	unknown_0:	Ü
	unknown_1:	Ü
	unknown_2:	Ü
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_63_layer_call_and_return_conditional_losses_70482542
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÜ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
 
_user_specified_nameinputs
á
×
8__inference_batch_normalization_64_layer_call_fn_7050376

inputs
unknown:	Ü
	unknown_0:	Ü
	unknown_1:	Ü
	unknown_2:	Ü
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_64_layer_call_and_return_conditional_losses_70484202
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÜ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
 
_user_specified_nameinputs

¶
S__inference_batch_normalization_63_layer_call_and_return_conditional_losses_7048192

inputs0
!batchnorm_readvariableop_resource:	Ü4
%batchnorm_mul_readvariableop_resource:	Ü2
#batchnorm_readvariableop_1_resource:	Ü2
#batchnorm_readvariableop_2_resource:	Ü
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Ü2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ü2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2

IdentityÂ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÜ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
 
_user_specified_nameinputs
²

ù
E__inference_dense_54_layer_call_and_return_conditional_losses_7050442

inputs2
matmul_readvariableop_resource:
ÜÈ.
biasadd_readvariableop_resource:	È
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ÜÈ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:È*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2	
BiasAddl
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÜ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
 
_user_specified_nameinputs

°
E__inference_dense_51_layer_call_and_return_conditional_losses_7050397

inputs2
matmul_readvariableop_resource:
ÜÜ
identity¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ÜÜ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
MatMull
IdentityIdentityMatMul:product:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:ÿÿÿÿÿÿÿÿÿÜ: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
 
_user_specified_nameinputs
Ýá

T__inference_feed_forward_sub_net_10_layer_call_and_return_conditional_losses_7049738
input_1M
>batch_normalization_60_assignmovingavg_readvariableop_resource:	ÈO
@batch_normalization_60_assignmovingavg_1_readvariableop_resource:	ÈK
<batch_normalization_60_batchnorm_mul_readvariableop_resource:	ÈG
8batch_normalization_60_batchnorm_readvariableop_resource:	È;
'dense_50_matmul_readvariableop_resource:
ÈÜM
>batch_normalization_61_assignmovingavg_readvariableop_resource:	ÜO
@batch_normalization_61_assignmovingavg_1_readvariableop_resource:	ÜK
<batch_normalization_61_batchnorm_mul_readvariableop_resource:	ÜG
8batch_normalization_61_batchnorm_readvariableop_resource:	Ü;
'dense_51_matmul_readvariableop_resource:
ÜÜM
>batch_normalization_62_assignmovingavg_readvariableop_resource:	ÜO
@batch_normalization_62_assignmovingavg_1_readvariableop_resource:	ÜK
<batch_normalization_62_batchnorm_mul_readvariableop_resource:	ÜG
8batch_normalization_62_batchnorm_readvariableop_resource:	Ü;
'dense_52_matmul_readvariableop_resource:
ÜÜM
>batch_normalization_63_assignmovingavg_readvariableop_resource:	ÜO
@batch_normalization_63_assignmovingavg_1_readvariableop_resource:	ÜK
<batch_normalization_63_batchnorm_mul_readvariableop_resource:	ÜG
8batch_normalization_63_batchnorm_readvariableop_resource:	Ü;
'dense_53_matmul_readvariableop_resource:
ÜÜM
>batch_normalization_64_assignmovingavg_readvariableop_resource:	ÜO
@batch_normalization_64_assignmovingavg_1_readvariableop_resource:	ÜK
<batch_normalization_64_batchnorm_mul_readvariableop_resource:	ÜG
8batch_normalization_64_batchnorm_readvariableop_resource:	Ü;
'dense_54_matmul_readvariableop_resource:
ÜÈ7
(dense_54_biasadd_readvariableop_resource:	È
identity¢&batch_normalization_60/AssignMovingAvg¢5batch_normalization_60/AssignMovingAvg/ReadVariableOp¢(batch_normalization_60/AssignMovingAvg_1¢7batch_normalization_60/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_60/batchnorm/ReadVariableOp¢3batch_normalization_60/batchnorm/mul/ReadVariableOp¢&batch_normalization_61/AssignMovingAvg¢5batch_normalization_61/AssignMovingAvg/ReadVariableOp¢(batch_normalization_61/AssignMovingAvg_1¢7batch_normalization_61/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_61/batchnorm/ReadVariableOp¢3batch_normalization_61/batchnorm/mul/ReadVariableOp¢&batch_normalization_62/AssignMovingAvg¢5batch_normalization_62/AssignMovingAvg/ReadVariableOp¢(batch_normalization_62/AssignMovingAvg_1¢7batch_normalization_62/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_62/batchnorm/ReadVariableOp¢3batch_normalization_62/batchnorm/mul/ReadVariableOp¢&batch_normalization_63/AssignMovingAvg¢5batch_normalization_63/AssignMovingAvg/ReadVariableOp¢(batch_normalization_63/AssignMovingAvg_1¢7batch_normalization_63/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_63/batchnorm/ReadVariableOp¢3batch_normalization_63/batchnorm/mul/ReadVariableOp¢&batch_normalization_64/AssignMovingAvg¢5batch_normalization_64/AssignMovingAvg/ReadVariableOp¢(batch_normalization_64/AssignMovingAvg_1¢7batch_normalization_64/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_64/batchnorm/ReadVariableOp¢3batch_normalization_64/batchnorm/mul/ReadVariableOp¢dense_50/MatMul/ReadVariableOp¢dense_51/MatMul/ReadVariableOp¢dense_52/MatMul/ReadVariableOp¢dense_53/MatMul/ReadVariableOp¢dense_54/BiasAdd/ReadVariableOp¢dense_54/MatMul/ReadVariableOp¸
5batch_normalization_60/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_60/moments/mean/reduction_indicesÖ
#batch_normalization_60/moments/meanMeaninput_1>batch_normalization_60/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	È*
	keep_dims(2%
#batch_normalization_60/moments/meanÂ
+batch_normalization_60/moments/StopGradientStopGradient,batch_normalization_60/moments/mean:output:0*
T0*
_output_shapes
:	È2-
+batch_normalization_60/moments/StopGradientë
0batch_normalization_60/moments/SquaredDifferenceSquaredDifferenceinput_14batch_normalization_60/moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ22
0batch_normalization_60/moments/SquaredDifferenceÀ
9batch_normalization_60/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_60/moments/variance/reduction_indices
'batch_normalization_60/moments/varianceMean4batch_normalization_60/moments/SquaredDifference:z:0Bbatch_normalization_60/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	È*
	keep_dims(2)
'batch_normalization_60/moments/varianceÆ
&batch_normalization_60/moments/SqueezeSqueeze,batch_normalization_60/moments/mean:output:0*
T0*
_output_shapes	
:È*
squeeze_dims
 2(
&batch_normalization_60/moments/SqueezeÎ
(batch_normalization_60/moments/Squeeze_1Squeeze0batch_normalization_60/moments/variance:output:0*
T0*
_output_shapes	
:È*
squeeze_dims
 2*
(batch_normalization_60/moments/Squeeze_1¡
,batch_normalization_60/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_60/AssignMovingAvg/decayÉ
+batch_normalization_60/AssignMovingAvg/CastCast5batch_normalization_60/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_60/AssignMovingAvg/Castê
5batch_normalization_60/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_60_assignmovingavg_readvariableop_resource*
_output_shapes	
:È*
dtype027
5batch_normalization_60/AssignMovingAvg/ReadVariableOpõ
*batch_normalization_60/AssignMovingAvg/subSub=batch_normalization_60/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_60/moments/Squeeze:output:0*
T0*
_output_shapes	
:È2,
*batch_normalization_60/AssignMovingAvg/subæ
*batch_normalization_60/AssignMovingAvg/mulMul.batch_normalization_60/AssignMovingAvg/sub:z:0/batch_normalization_60/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes	
:È2,
*batch_normalization_60/AssignMovingAvg/mul²
&batch_normalization_60/AssignMovingAvgAssignSubVariableOp>batch_normalization_60_assignmovingavg_readvariableop_resource.batch_normalization_60/AssignMovingAvg/mul:z:06^batch_normalization_60/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_60/AssignMovingAvg¥
.batch_normalization_60/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_60/AssignMovingAvg_1/decayÏ
-batch_normalization_60/AssignMovingAvg_1/CastCast7batch_normalization_60/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_60/AssignMovingAvg_1/Castð
7batch_normalization_60/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_60_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:È*
dtype029
7batch_normalization_60/AssignMovingAvg_1/ReadVariableOpý
,batch_normalization_60/AssignMovingAvg_1/subSub?batch_normalization_60/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_60/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:È2.
,batch_normalization_60/AssignMovingAvg_1/subî
,batch_normalization_60/AssignMovingAvg_1/mulMul0batch_normalization_60/AssignMovingAvg_1/sub:z:01batch_normalization_60/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes	
:È2.
,batch_normalization_60/AssignMovingAvg_1/mul¼
(batch_normalization_60/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_60_assignmovingavg_1_readvariableop_resource0batch_normalization_60/AssignMovingAvg_1/mul:z:08^batch_normalization_60/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_60/AssignMovingAvg_1
&batch_normalization_60/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_60/batchnorm/add/yß
$batch_normalization_60/batchnorm/addAddV21batch_normalization_60/moments/Squeeze_1:output:0/batch_normalization_60/batchnorm/add/y:output:0*
T0*
_output_shapes	
:È2&
$batch_normalization_60/batchnorm/add©
&batch_normalization_60/batchnorm/RsqrtRsqrt(batch_normalization_60/batchnorm/add:z:0*
T0*
_output_shapes	
:È2(
&batch_normalization_60/batchnorm/Rsqrtä
3batch_normalization_60/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_60_batchnorm_mul_readvariableop_resource*
_output_shapes	
:È*
dtype025
3batch_normalization_60/batchnorm/mul/ReadVariableOpâ
$batch_normalization_60/batchnorm/mulMul*batch_normalization_60/batchnorm/Rsqrt:y:0;batch_normalization_60/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:È2&
$batch_normalization_60/batchnorm/mul½
&batch_normalization_60/batchnorm/mul_1Mulinput_1(batch_normalization_60/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2(
&batch_normalization_60/batchnorm/mul_1Ø
&batch_normalization_60/batchnorm/mul_2Mul/batch_normalization_60/moments/Squeeze:output:0(batch_normalization_60/batchnorm/mul:z:0*
T0*
_output_shapes	
:È2(
&batch_normalization_60/batchnorm/mul_2Ø
/batch_normalization_60/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_60_batchnorm_readvariableop_resource*
_output_shapes	
:È*
dtype021
/batch_normalization_60/batchnorm/ReadVariableOpÞ
$batch_normalization_60/batchnorm/subSub7batch_normalization_60/batchnorm/ReadVariableOp:value:0*batch_normalization_60/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:È2&
$batch_normalization_60/batchnorm/subâ
&batch_normalization_60/batchnorm/add_1AddV2*batch_normalization_60/batchnorm/mul_1:z:0(batch_normalization_60/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2(
&batch_normalization_60/batchnorm/add_1ª
dense_50/MatMul/ReadVariableOpReadVariableOp'dense_50_matmul_readvariableop_resource* 
_output_shapes
:
ÈÜ*
dtype02 
dense_50/MatMul/ReadVariableOp³
dense_50/MatMulMatMul*batch_normalization_60/batchnorm/add_1:z:0&dense_50/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
dense_50/MatMul¸
5batch_normalization_61/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_61/moments/mean/reduction_indicesè
#batch_normalization_61/moments/meanMeandense_50/MatMul:product:0>batch_normalization_61/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	Ü*
	keep_dims(2%
#batch_normalization_61/moments/meanÂ
+batch_normalization_61/moments/StopGradientStopGradient,batch_normalization_61/moments/mean:output:0*
T0*
_output_shapes
:	Ü2-
+batch_normalization_61/moments/StopGradientý
0batch_normalization_61/moments/SquaredDifferenceSquaredDifferencedense_50/MatMul:product:04batch_normalization_61/moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ22
0batch_normalization_61/moments/SquaredDifferenceÀ
9batch_normalization_61/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_61/moments/variance/reduction_indices
'batch_normalization_61/moments/varianceMean4batch_normalization_61/moments/SquaredDifference:z:0Bbatch_normalization_61/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	Ü*
	keep_dims(2)
'batch_normalization_61/moments/varianceÆ
&batch_normalization_61/moments/SqueezeSqueeze,batch_normalization_61/moments/mean:output:0*
T0*
_output_shapes	
:Ü*
squeeze_dims
 2(
&batch_normalization_61/moments/SqueezeÎ
(batch_normalization_61/moments/Squeeze_1Squeeze0batch_normalization_61/moments/variance:output:0*
T0*
_output_shapes	
:Ü*
squeeze_dims
 2*
(batch_normalization_61/moments/Squeeze_1¡
,batch_normalization_61/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_61/AssignMovingAvg/decayÉ
+batch_normalization_61/AssignMovingAvg/CastCast5batch_normalization_61/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_61/AssignMovingAvg/Castê
5batch_normalization_61/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_61_assignmovingavg_readvariableop_resource*
_output_shapes	
:Ü*
dtype027
5batch_normalization_61/AssignMovingAvg/ReadVariableOpõ
*batch_normalization_61/AssignMovingAvg/subSub=batch_normalization_61/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_61/moments/Squeeze:output:0*
T0*
_output_shapes	
:Ü2,
*batch_normalization_61/AssignMovingAvg/subæ
*batch_normalization_61/AssignMovingAvg/mulMul.batch_normalization_61/AssignMovingAvg/sub:z:0/batch_normalization_61/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes	
:Ü2,
*batch_normalization_61/AssignMovingAvg/mul²
&batch_normalization_61/AssignMovingAvgAssignSubVariableOp>batch_normalization_61_assignmovingavg_readvariableop_resource.batch_normalization_61/AssignMovingAvg/mul:z:06^batch_normalization_61/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_61/AssignMovingAvg¥
.batch_normalization_61/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_61/AssignMovingAvg_1/decayÏ
-batch_normalization_61/AssignMovingAvg_1/CastCast7batch_normalization_61/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_61/AssignMovingAvg_1/Castð
7batch_normalization_61/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_61_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:Ü*
dtype029
7batch_normalization_61/AssignMovingAvg_1/ReadVariableOpý
,batch_normalization_61/AssignMovingAvg_1/subSub?batch_normalization_61/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_61/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:Ü2.
,batch_normalization_61/AssignMovingAvg_1/subî
,batch_normalization_61/AssignMovingAvg_1/mulMul0batch_normalization_61/AssignMovingAvg_1/sub:z:01batch_normalization_61/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes	
:Ü2.
,batch_normalization_61/AssignMovingAvg_1/mul¼
(batch_normalization_61/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_61_assignmovingavg_1_readvariableop_resource0batch_normalization_61/AssignMovingAvg_1/mul:z:08^batch_normalization_61/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_61/AssignMovingAvg_1
&batch_normalization_61/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_61/batchnorm/add/yß
$batch_normalization_61/batchnorm/addAddV21batch_normalization_61/moments/Squeeze_1:output:0/batch_normalization_61/batchnorm/add/y:output:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_61/batchnorm/add©
&batch_normalization_61/batchnorm/RsqrtRsqrt(batch_normalization_61/batchnorm/add:z:0*
T0*
_output_shapes	
:Ü2(
&batch_normalization_61/batchnorm/Rsqrtä
3batch_normalization_61/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_61_batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ü*
dtype025
3batch_normalization_61/batchnorm/mul/ReadVariableOpâ
$batch_normalization_61/batchnorm/mulMul*batch_normalization_61/batchnorm/Rsqrt:y:0;batch_normalization_61/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_61/batchnorm/mulÏ
&batch_normalization_61/batchnorm/mul_1Muldense_50/MatMul:product:0(batch_normalization_61/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2(
&batch_normalization_61/batchnorm/mul_1Ø
&batch_normalization_61/batchnorm/mul_2Mul/batch_normalization_61/moments/Squeeze:output:0(batch_normalization_61/batchnorm/mul:z:0*
T0*
_output_shapes	
:Ü2(
&batch_normalization_61/batchnorm/mul_2Ø
/batch_normalization_61/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_61_batchnorm_readvariableop_resource*
_output_shapes	
:Ü*
dtype021
/batch_normalization_61/batchnorm/ReadVariableOpÞ
$batch_normalization_61/batchnorm/subSub7batch_normalization_61/batchnorm/ReadVariableOp:value:0*batch_normalization_61/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_61/batchnorm/subâ
&batch_normalization_61/batchnorm/add_1AddV2*batch_normalization_61/batchnorm/mul_1:z:0(batch_normalization_61/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2(
&batch_normalization_61/batchnorm/add_1s
ReluRelu*batch_normalization_61/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
Reluª
dense_51/MatMul/ReadVariableOpReadVariableOp'dense_51_matmul_readvariableop_resource* 
_output_shapes
:
ÜÜ*
dtype02 
dense_51/MatMul/ReadVariableOp
dense_51/MatMulMatMulRelu:activations:0&dense_51/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
dense_51/MatMul¸
5batch_normalization_62/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_62/moments/mean/reduction_indicesè
#batch_normalization_62/moments/meanMeandense_51/MatMul:product:0>batch_normalization_62/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	Ü*
	keep_dims(2%
#batch_normalization_62/moments/meanÂ
+batch_normalization_62/moments/StopGradientStopGradient,batch_normalization_62/moments/mean:output:0*
T0*
_output_shapes
:	Ü2-
+batch_normalization_62/moments/StopGradientý
0batch_normalization_62/moments/SquaredDifferenceSquaredDifferencedense_51/MatMul:product:04batch_normalization_62/moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ22
0batch_normalization_62/moments/SquaredDifferenceÀ
9batch_normalization_62/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_62/moments/variance/reduction_indices
'batch_normalization_62/moments/varianceMean4batch_normalization_62/moments/SquaredDifference:z:0Bbatch_normalization_62/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	Ü*
	keep_dims(2)
'batch_normalization_62/moments/varianceÆ
&batch_normalization_62/moments/SqueezeSqueeze,batch_normalization_62/moments/mean:output:0*
T0*
_output_shapes	
:Ü*
squeeze_dims
 2(
&batch_normalization_62/moments/SqueezeÎ
(batch_normalization_62/moments/Squeeze_1Squeeze0batch_normalization_62/moments/variance:output:0*
T0*
_output_shapes	
:Ü*
squeeze_dims
 2*
(batch_normalization_62/moments/Squeeze_1¡
,batch_normalization_62/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_62/AssignMovingAvg/decayÉ
+batch_normalization_62/AssignMovingAvg/CastCast5batch_normalization_62/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_62/AssignMovingAvg/Castê
5batch_normalization_62/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_62_assignmovingavg_readvariableop_resource*
_output_shapes	
:Ü*
dtype027
5batch_normalization_62/AssignMovingAvg/ReadVariableOpõ
*batch_normalization_62/AssignMovingAvg/subSub=batch_normalization_62/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_62/moments/Squeeze:output:0*
T0*
_output_shapes	
:Ü2,
*batch_normalization_62/AssignMovingAvg/subæ
*batch_normalization_62/AssignMovingAvg/mulMul.batch_normalization_62/AssignMovingAvg/sub:z:0/batch_normalization_62/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes	
:Ü2,
*batch_normalization_62/AssignMovingAvg/mul²
&batch_normalization_62/AssignMovingAvgAssignSubVariableOp>batch_normalization_62_assignmovingavg_readvariableop_resource.batch_normalization_62/AssignMovingAvg/mul:z:06^batch_normalization_62/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_62/AssignMovingAvg¥
.batch_normalization_62/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_62/AssignMovingAvg_1/decayÏ
-batch_normalization_62/AssignMovingAvg_1/CastCast7batch_normalization_62/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_62/AssignMovingAvg_1/Castð
7batch_normalization_62/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_62_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:Ü*
dtype029
7batch_normalization_62/AssignMovingAvg_1/ReadVariableOpý
,batch_normalization_62/AssignMovingAvg_1/subSub?batch_normalization_62/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_62/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:Ü2.
,batch_normalization_62/AssignMovingAvg_1/subî
,batch_normalization_62/AssignMovingAvg_1/mulMul0batch_normalization_62/AssignMovingAvg_1/sub:z:01batch_normalization_62/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes	
:Ü2.
,batch_normalization_62/AssignMovingAvg_1/mul¼
(batch_normalization_62/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_62_assignmovingavg_1_readvariableop_resource0batch_normalization_62/AssignMovingAvg_1/mul:z:08^batch_normalization_62/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_62/AssignMovingAvg_1
&batch_normalization_62/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_62/batchnorm/add/yß
$batch_normalization_62/batchnorm/addAddV21batch_normalization_62/moments/Squeeze_1:output:0/batch_normalization_62/batchnorm/add/y:output:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_62/batchnorm/add©
&batch_normalization_62/batchnorm/RsqrtRsqrt(batch_normalization_62/batchnorm/add:z:0*
T0*
_output_shapes	
:Ü2(
&batch_normalization_62/batchnorm/Rsqrtä
3batch_normalization_62/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_62_batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ü*
dtype025
3batch_normalization_62/batchnorm/mul/ReadVariableOpâ
$batch_normalization_62/batchnorm/mulMul*batch_normalization_62/batchnorm/Rsqrt:y:0;batch_normalization_62/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_62/batchnorm/mulÏ
&batch_normalization_62/batchnorm/mul_1Muldense_51/MatMul:product:0(batch_normalization_62/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2(
&batch_normalization_62/batchnorm/mul_1Ø
&batch_normalization_62/batchnorm/mul_2Mul/batch_normalization_62/moments/Squeeze:output:0(batch_normalization_62/batchnorm/mul:z:0*
T0*
_output_shapes	
:Ü2(
&batch_normalization_62/batchnorm/mul_2Ø
/batch_normalization_62/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_62_batchnorm_readvariableop_resource*
_output_shapes	
:Ü*
dtype021
/batch_normalization_62/batchnorm/ReadVariableOpÞ
$batch_normalization_62/batchnorm/subSub7batch_normalization_62/batchnorm/ReadVariableOp:value:0*batch_normalization_62/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_62/batchnorm/subâ
&batch_normalization_62/batchnorm/add_1AddV2*batch_normalization_62/batchnorm/mul_1:z:0(batch_normalization_62/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2(
&batch_normalization_62/batchnorm/add_1w
Relu_1Relu*batch_normalization_62/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
Relu_1ª
dense_52/MatMul/ReadVariableOpReadVariableOp'dense_52_matmul_readvariableop_resource* 
_output_shapes
:
ÜÜ*
dtype02 
dense_52/MatMul/ReadVariableOp
dense_52/MatMulMatMulRelu_1:activations:0&dense_52/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
dense_52/MatMul¸
5batch_normalization_63/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_63/moments/mean/reduction_indicesè
#batch_normalization_63/moments/meanMeandense_52/MatMul:product:0>batch_normalization_63/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	Ü*
	keep_dims(2%
#batch_normalization_63/moments/meanÂ
+batch_normalization_63/moments/StopGradientStopGradient,batch_normalization_63/moments/mean:output:0*
T0*
_output_shapes
:	Ü2-
+batch_normalization_63/moments/StopGradientý
0batch_normalization_63/moments/SquaredDifferenceSquaredDifferencedense_52/MatMul:product:04batch_normalization_63/moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ22
0batch_normalization_63/moments/SquaredDifferenceÀ
9batch_normalization_63/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_63/moments/variance/reduction_indices
'batch_normalization_63/moments/varianceMean4batch_normalization_63/moments/SquaredDifference:z:0Bbatch_normalization_63/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	Ü*
	keep_dims(2)
'batch_normalization_63/moments/varianceÆ
&batch_normalization_63/moments/SqueezeSqueeze,batch_normalization_63/moments/mean:output:0*
T0*
_output_shapes	
:Ü*
squeeze_dims
 2(
&batch_normalization_63/moments/SqueezeÎ
(batch_normalization_63/moments/Squeeze_1Squeeze0batch_normalization_63/moments/variance:output:0*
T0*
_output_shapes	
:Ü*
squeeze_dims
 2*
(batch_normalization_63/moments/Squeeze_1¡
,batch_normalization_63/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_63/AssignMovingAvg/decayÉ
+batch_normalization_63/AssignMovingAvg/CastCast5batch_normalization_63/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_63/AssignMovingAvg/Castê
5batch_normalization_63/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_63_assignmovingavg_readvariableop_resource*
_output_shapes	
:Ü*
dtype027
5batch_normalization_63/AssignMovingAvg/ReadVariableOpõ
*batch_normalization_63/AssignMovingAvg/subSub=batch_normalization_63/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_63/moments/Squeeze:output:0*
T0*
_output_shapes	
:Ü2,
*batch_normalization_63/AssignMovingAvg/subæ
*batch_normalization_63/AssignMovingAvg/mulMul.batch_normalization_63/AssignMovingAvg/sub:z:0/batch_normalization_63/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes	
:Ü2,
*batch_normalization_63/AssignMovingAvg/mul²
&batch_normalization_63/AssignMovingAvgAssignSubVariableOp>batch_normalization_63_assignmovingavg_readvariableop_resource.batch_normalization_63/AssignMovingAvg/mul:z:06^batch_normalization_63/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_63/AssignMovingAvg¥
.batch_normalization_63/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_63/AssignMovingAvg_1/decayÏ
-batch_normalization_63/AssignMovingAvg_1/CastCast7batch_normalization_63/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_63/AssignMovingAvg_1/Castð
7batch_normalization_63/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_63_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:Ü*
dtype029
7batch_normalization_63/AssignMovingAvg_1/ReadVariableOpý
,batch_normalization_63/AssignMovingAvg_1/subSub?batch_normalization_63/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_63/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:Ü2.
,batch_normalization_63/AssignMovingAvg_1/subî
,batch_normalization_63/AssignMovingAvg_1/mulMul0batch_normalization_63/AssignMovingAvg_1/sub:z:01batch_normalization_63/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes	
:Ü2.
,batch_normalization_63/AssignMovingAvg_1/mul¼
(batch_normalization_63/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_63_assignmovingavg_1_readvariableop_resource0batch_normalization_63/AssignMovingAvg_1/mul:z:08^batch_normalization_63/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_63/AssignMovingAvg_1
&batch_normalization_63/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_63/batchnorm/add/yß
$batch_normalization_63/batchnorm/addAddV21batch_normalization_63/moments/Squeeze_1:output:0/batch_normalization_63/batchnorm/add/y:output:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_63/batchnorm/add©
&batch_normalization_63/batchnorm/RsqrtRsqrt(batch_normalization_63/batchnorm/add:z:0*
T0*
_output_shapes	
:Ü2(
&batch_normalization_63/batchnorm/Rsqrtä
3batch_normalization_63/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_63_batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ü*
dtype025
3batch_normalization_63/batchnorm/mul/ReadVariableOpâ
$batch_normalization_63/batchnorm/mulMul*batch_normalization_63/batchnorm/Rsqrt:y:0;batch_normalization_63/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_63/batchnorm/mulÏ
&batch_normalization_63/batchnorm/mul_1Muldense_52/MatMul:product:0(batch_normalization_63/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2(
&batch_normalization_63/batchnorm/mul_1Ø
&batch_normalization_63/batchnorm/mul_2Mul/batch_normalization_63/moments/Squeeze:output:0(batch_normalization_63/batchnorm/mul:z:0*
T0*
_output_shapes	
:Ü2(
&batch_normalization_63/batchnorm/mul_2Ø
/batch_normalization_63/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_63_batchnorm_readvariableop_resource*
_output_shapes	
:Ü*
dtype021
/batch_normalization_63/batchnorm/ReadVariableOpÞ
$batch_normalization_63/batchnorm/subSub7batch_normalization_63/batchnorm/ReadVariableOp:value:0*batch_normalization_63/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_63/batchnorm/subâ
&batch_normalization_63/batchnorm/add_1AddV2*batch_normalization_63/batchnorm/mul_1:z:0(batch_normalization_63/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2(
&batch_normalization_63/batchnorm/add_1w
Relu_2Relu*batch_normalization_63/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
Relu_2ª
dense_53/MatMul/ReadVariableOpReadVariableOp'dense_53_matmul_readvariableop_resource* 
_output_shapes
:
ÜÜ*
dtype02 
dense_53/MatMul/ReadVariableOp
dense_53/MatMulMatMulRelu_2:activations:0&dense_53/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
dense_53/MatMul¸
5batch_normalization_64/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_64/moments/mean/reduction_indicesè
#batch_normalization_64/moments/meanMeandense_53/MatMul:product:0>batch_normalization_64/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	Ü*
	keep_dims(2%
#batch_normalization_64/moments/meanÂ
+batch_normalization_64/moments/StopGradientStopGradient,batch_normalization_64/moments/mean:output:0*
T0*
_output_shapes
:	Ü2-
+batch_normalization_64/moments/StopGradientý
0batch_normalization_64/moments/SquaredDifferenceSquaredDifferencedense_53/MatMul:product:04batch_normalization_64/moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ22
0batch_normalization_64/moments/SquaredDifferenceÀ
9batch_normalization_64/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_64/moments/variance/reduction_indices
'batch_normalization_64/moments/varianceMean4batch_normalization_64/moments/SquaredDifference:z:0Bbatch_normalization_64/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	Ü*
	keep_dims(2)
'batch_normalization_64/moments/varianceÆ
&batch_normalization_64/moments/SqueezeSqueeze,batch_normalization_64/moments/mean:output:0*
T0*
_output_shapes	
:Ü*
squeeze_dims
 2(
&batch_normalization_64/moments/SqueezeÎ
(batch_normalization_64/moments/Squeeze_1Squeeze0batch_normalization_64/moments/variance:output:0*
T0*
_output_shapes	
:Ü*
squeeze_dims
 2*
(batch_normalization_64/moments/Squeeze_1¡
,batch_normalization_64/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_64/AssignMovingAvg/decayÉ
+batch_normalization_64/AssignMovingAvg/CastCast5batch_normalization_64/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_64/AssignMovingAvg/Castê
5batch_normalization_64/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_64_assignmovingavg_readvariableop_resource*
_output_shapes	
:Ü*
dtype027
5batch_normalization_64/AssignMovingAvg/ReadVariableOpõ
*batch_normalization_64/AssignMovingAvg/subSub=batch_normalization_64/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_64/moments/Squeeze:output:0*
T0*
_output_shapes	
:Ü2,
*batch_normalization_64/AssignMovingAvg/subæ
*batch_normalization_64/AssignMovingAvg/mulMul.batch_normalization_64/AssignMovingAvg/sub:z:0/batch_normalization_64/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes	
:Ü2,
*batch_normalization_64/AssignMovingAvg/mul²
&batch_normalization_64/AssignMovingAvgAssignSubVariableOp>batch_normalization_64_assignmovingavg_readvariableop_resource.batch_normalization_64/AssignMovingAvg/mul:z:06^batch_normalization_64/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_64/AssignMovingAvg¥
.batch_normalization_64/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_64/AssignMovingAvg_1/decayÏ
-batch_normalization_64/AssignMovingAvg_1/CastCast7batch_normalization_64/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_64/AssignMovingAvg_1/Castð
7batch_normalization_64/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_64_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:Ü*
dtype029
7batch_normalization_64/AssignMovingAvg_1/ReadVariableOpý
,batch_normalization_64/AssignMovingAvg_1/subSub?batch_normalization_64/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_64/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:Ü2.
,batch_normalization_64/AssignMovingAvg_1/subî
,batch_normalization_64/AssignMovingAvg_1/mulMul0batch_normalization_64/AssignMovingAvg_1/sub:z:01batch_normalization_64/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes	
:Ü2.
,batch_normalization_64/AssignMovingAvg_1/mul¼
(batch_normalization_64/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_64_assignmovingavg_1_readvariableop_resource0batch_normalization_64/AssignMovingAvg_1/mul:z:08^batch_normalization_64/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_64/AssignMovingAvg_1
&batch_normalization_64/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_64/batchnorm/add/yß
$batch_normalization_64/batchnorm/addAddV21batch_normalization_64/moments/Squeeze_1:output:0/batch_normalization_64/batchnorm/add/y:output:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_64/batchnorm/add©
&batch_normalization_64/batchnorm/RsqrtRsqrt(batch_normalization_64/batchnorm/add:z:0*
T0*
_output_shapes	
:Ü2(
&batch_normalization_64/batchnorm/Rsqrtä
3batch_normalization_64/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_64_batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ü*
dtype025
3batch_normalization_64/batchnorm/mul/ReadVariableOpâ
$batch_normalization_64/batchnorm/mulMul*batch_normalization_64/batchnorm/Rsqrt:y:0;batch_normalization_64/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_64/batchnorm/mulÏ
&batch_normalization_64/batchnorm/mul_1Muldense_53/MatMul:product:0(batch_normalization_64/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2(
&batch_normalization_64/batchnorm/mul_1Ø
&batch_normalization_64/batchnorm/mul_2Mul/batch_normalization_64/moments/Squeeze:output:0(batch_normalization_64/batchnorm/mul:z:0*
T0*
_output_shapes	
:Ü2(
&batch_normalization_64/batchnorm/mul_2Ø
/batch_normalization_64/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_64_batchnorm_readvariableop_resource*
_output_shapes	
:Ü*
dtype021
/batch_normalization_64/batchnorm/ReadVariableOpÞ
$batch_normalization_64/batchnorm/subSub7batch_normalization_64/batchnorm/ReadVariableOp:value:0*batch_normalization_64/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_64/batchnorm/subâ
&batch_normalization_64/batchnorm/add_1AddV2*batch_normalization_64/batchnorm/mul_1:z:0(batch_normalization_64/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2(
&batch_normalization_64/batchnorm/add_1w
Relu_3Relu*batch_normalization_64/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
Relu_3ª
dense_54/MatMul/ReadVariableOpReadVariableOp'dense_54_matmul_readvariableop_resource* 
_output_shapes
:
ÜÈ*
dtype02 
dense_54/MatMul/ReadVariableOp
dense_54/MatMulMatMulRelu_3:activations:0&dense_54/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
dense_54/MatMul¨
dense_54/BiasAdd/ReadVariableOpReadVariableOp(dense_54_biasadd_readvariableop_resource*
_output_shapes	
:È*
dtype02!
dense_54/BiasAdd/ReadVariableOp¦
dense_54/BiasAddBiasAdddense_54/MatMul:product:0'dense_54/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
dense_54/BiasAddu
IdentityIdentitydense_54/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identityû
NoOpNoOp'^batch_normalization_60/AssignMovingAvg6^batch_normalization_60/AssignMovingAvg/ReadVariableOp)^batch_normalization_60/AssignMovingAvg_18^batch_normalization_60/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_60/batchnorm/ReadVariableOp4^batch_normalization_60/batchnorm/mul/ReadVariableOp'^batch_normalization_61/AssignMovingAvg6^batch_normalization_61/AssignMovingAvg/ReadVariableOp)^batch_normalization_61/AssignMovingAvg_18^batch_normalization_61/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_61/batchnorm/ReadVariableOp4^batch_normalization_61/batchnorm/mul/ReadVariableOp'^batch_normalization_62/AssignMovingAvg6^batch_normalization_62/AssignMovingAvg/ReadVariableOp)^batch_normalization_62/AssignMovingAvg_18^batch_normalization_62/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_62/batchnorm/ReadVariableOp4^batch_normalization_62/batchnorm/mul/ReadVariableOp'^batch_normalization_63/AssignMovingAvg6^batch_normalization_63/AssignMovingAvg/ReadVariableOp)^batch_normalization_63/AssignMovingAvg_18^batch_normalization_63/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_63/batchnorm/ReadVariableOp4^batch_normalization_63/batchnorm/mul/ReadVariableOp'^batch_normalization_64/AssignMovingAvg6^batch_normalization_64/AssignMovingAvg/ReadVariableOp)^batch_normalization_64/AssignMovingAvg_18^batch_normalization_64/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_64/batchnorm/ReadVariableOp4^batch_normalization_64/batchnorm/mul/ReadVariableOp^dense_50/MatMul/ReadVariableOp^dense_51/MatMul/ReadVariableOp^dense_52/MatMul/ReadVariableOp^dense_53/MatMul/ReadVariableOp ^dense_54/BiasAdd/ReadVariableOp^dense_54/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:ÿÿÿÿÿÿÿÿÿÈ: : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_60/AssignMovingAvg&batch_normalization_60/AssignMovingAvg2n
5batch_normalization_60/AssignMovingAvg/ReadVariableOp5batch_normalization_60/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_60/AssignMovingAvg_1(batch_normalization_60/AssignMovingAvg_12r
7batch_normalization_60/AssignMovingAvg_1/ReadVariableOp7batch_normalization_60/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_60/batchnorm/ReadVariableOp/batch_normalization_60/batchnorm/ReadVariableOp2j
3batch_normalization_60/batchnorm/mul/ReadVariableOp3batch_normalization_60/batchnorm/mul/ReadVariableOp2P
&batch_normalization_61/AssignMovingAvg&batch_normalization_61/AssignMovingAvg2n
5batch_normalization_61/AssignMovingAvg/ReadVariableOp5batch_normalization_61/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_61/AssignMovingAvg_1(batch_normalization_61/AssignMovingAvg_12r
7batch_normalization_61/AssignMovingAvg_1/ReadVariableOp7batch_normalization_61/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_61/batchnorm/ReadVariableOp/batch_normalization_61/batchnorm/ReadVariableOp2j
3batch_normalization_61/batchnorm/mul/ReadVariableOp3batch_normalization_61/batchnorm/mul/ReadVariableOp2P
&batch_normalization_62/AssignMovingAvg&batch_normalization_62/AssignMovingAvg2n
5batch_normalization_62/AssignMovingAvg/ReadVariableOp5batch_normalization_62/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_62/AssignMovingAvg_1(batch_normalization_62/AssignMovingAvg_12r
7batch_normalization_62/AssignMovingAvg_1/ReadVariableOp7batch_normalization_62/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_62/batchnorm/ReadVariableOp/batch_normalization_62/batchnorm/ReadVariableOp2j
3batch_normalization_62/batchnorm/mul/ReadVariableOp3batch_normalization_62/batchnorm/mul/ReadVariableOp2P
&batch_normalization_63/AssignMovingAvg&batch_normalization_63/AssignMovingAvg2n
5batch_normalization_63/AssignMovingAvg/ReadVariableOp5batch_normalization_63/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_63/AssignMovingAvg_1(batch_normalization_63/AssignMovingAvg_12r
7batch_normalization_63/AssignMovingAvg_1/ReadVariableOp7batch_normalization_63/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_63/batchnorm/ReadVariableOp/batch_normalization_63/batchnorm/ReadVariableOp2j
3batch_normalization_63/batchnorm/mul/ReadVariableOp3batch_normalization_63/batchnorm/mul/ReadVariableOp2P
&batch_normalization_64/AssignMovingAvg&batch_normalization_64/AssignMovingAvg2n
5batch_normalization_64/AssignMovingAvg/ReadVariableOp5batch_normalization_64/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_64/AssignMovingAvg_1(batch_normalization_64/AssignMovingAvg_12r
7batch_normalization_64/AssignMovingAvg_1/ReadVariableOp7batch_normalization_64/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_64/batchnorm/ReadVariableOp/batch_normalization_64/batchnorm/ReadVariableOp2j
3batch_normalization_64/batchnorm/mul/ReadVariableOp3batch_normalization_64/batchnorm/mul/ReadVariableOp2@
dense_50/MatMul/ReadVariableOpdense_50/MatMul/ReadVariableOp2@
dense_51/MatMul/ReadVariableOpdense_51/MatMul/ReadVariableOp2@
dense_52/MatMul/ReadVariableOpdense_52/MatMul/ReadVariableOp2@
dense_53/MatMul/ReadVariableOpdense_53/MatMul/ReadVariableOp2B
dense_54/BiasAdd/ReadVariableOpdense_54/BiasAdd/ReadVariableOp2@
dense_54/MatMul/ReadVariableOpdense_54/MatMul/ReadVariableOp:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
!
_user_specified_name	input_1
¥
½
9__inference_feed_forward_sub_net_10_layer_call_fn_7049795
input_1
unknown:	È
	unknown_0:	È
	unknown_1:	È
	unknown_2:	È
	unknown_3:
ÈÜ
	unknown_4:	Ü
	unknown_5:	Ü
	unknown_6:	Ü
	unknown_7:	Ü
	unknown_8:
ÜÜ
	unknown_9:	Ü

unknown_10:	Ü

unknown_11:	Ü

unknown_12:	Ü

unknown_13:
ÜÜ

unknown_14:	Ü

unknown_15:	Ü

unknown_16:	Ü

unknown_17:	Ü

unknown_18:
ÜÜ

unknown_19:	Ü

unknown_20:	Ü

unknown_21:	Ü

unknown_22:	Ü

unknown_23:
ÜÈ

unknown_24:	È
identity¢StatefulPartitionedCallÍ
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
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_feed_forward_sub_net_10_layer_call_and_return_conditional_losses_70486172
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:ÿÿÿÿÿÿÿÿÿÈ: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
!
_user_specified_name	input_1

¶
S__inference_batch_normalization_63_layer_call_and_return_conditional_losses_7050232

inputs0
!batchnorm_readvariableop_resource:	Ü4
%batchnorm_mul_readvariableop_resource:	Ü2
#batchnorm_readvariableop_1_resource:	Ü2
#batchnorm_readvariableop_2_resource:	Ü
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Ü2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ü2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2

IdentityÂ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÜ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
 
_user_specified_nameinputs
ö,
ð
S__inference_batch_normalization_60_layer_call_and_return_conditional_losses_7050022

inputs6
'assignmovingavg_readvariableop_resource:	È8
)assignmovingavg_1_readvariableop_resource:	È4
%batchnorm_mul_readvariableop_resource:	È0
!batchnorm_readvariableop_resource:	È
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	È*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	È2
moments/StopGradient¥
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices³
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	È*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:È*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:È*
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
AssignMovingAvg/Cast¥
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:È*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:È2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes	
:È2
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
AssignMovingAvg_1/Cast«
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:È*
dtype02"
 AssignMovingAvg_1/ReadVariableOp¡
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:È2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes	
:È2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1k
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:È2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:È2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:È*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:È2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:È2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:È*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:È2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identityò
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs

½
9__inference_feed_forward_sub_net_10_layer_call_fn_7049966
input_1
unknown:	È
	unknown_0:	È
	unknown_1:	È
	unknown_2:	È
	unknown_3:
ÈÜ
	unknown_4:	Ü
	unknown_5:	Ü
	unknown_6:	Ü
	unknown_7:	Ü
	unknown_8:
ÜÜ
	unknown_9:	Ü

unknown_10:	Ü

unknown_11:	Ü

unknown_12:	Ü

unknown_13:
ÜÜ

unknown_14:	Ü

unknown_15:	Ü

unknown_16:	Ü

unknown_17:	Ü

unknown_18:
ÜÜ

unknown_19:	Ü

unknown_20:	Ü

unknown_21:	Ü

unknown_22:	Ü

unknown_23:
ÜÈ

unknown_24:	È
identity¢StatefulPartitionedCallÃ
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
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_feed_forward_sub_net_10_layer_call_and_return_conditional_losses_70488432
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:ÿÿÿÿÿÿÿÿÿÈ: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
!
_user_specified_name	input_1
ÄE

T__inference_feed_forward_sub_net_10_layer_call_and_return_conditional_losses_7048843
x-
batch_normalization_60_7048776:	È-
batch_normalization_60_7048778:	È-
batch_normalization_60_7048780:	È-
batch_normalization_60_7048782:	È$
dense_50_7048785:
ÈÜ-
batch_normalization_61_7048788:	Ü-
batch_normalization_61_7048790:	Ü-
batch_normalization_61_7048792:	Ü-
batch_normalization_61_7048794:	Ü$
dense_51_7048798:
ÜÜ-
batch_normalization_62_7048801:	Ü-
batch_normalization_62_7048803:	Ü-
batch_normalization_62_7048805:	Ü-
batch_normalization_62_7048807:	Ü$
dense_52_7048811:
ÜÜ-
batch_normalization_63_7048814:	Ü-
batch_normalization_63_7048816:	Ü-
batch_normalization_63_7048818:	Ü-
batch_normalization_63_7048820:	Ü$
dense_53_7048824:
ÜÜ-
batch_normalization_64_7048827:	Ü-
batch_normalization_64_7048829:	Ü-
batch_normalization_64_7048831:	Ü-
batch_normalization_64_7048833:	Ü$
dense_54_7048837:
ÜÈ
dense_54_7048839:	È
identity¢.batch_normalization_60/StatefulPartitionedCall¢.batch_normalization_61/StatefulPartitionedCall¢.batch_normalization_62/StatefulPartitionedCall¢.batch_normalization_63/StatefulPartitionedCall¢.batch_normalization_64/StatefulPartitionedCall¢ dense_50/StatefulPartitionedCall¢ dense_51/StatefulPartitionedCall¢ dense_52/StatefulPartitionedCall¢ dense_53/StatefulPartitionedCall¢ dense_54/StatefulPartitionedCall
.batch_normalization_60/StatefulPartitionedCallStatefulPartitionedCallxbatch_normalization_60_7048776batch_normalization_60_7048778batch_normalization_60_7048780batch_normalization_60_7048782*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_60_layer_call_and_return_conditional_losses_704775620
.batch_normalization_60/StatefulPartitionedCallµ
 dense_50/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_60/StatefulPartitionedCall:output:0dense_50_7048785*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_50_layer_call_and_return_conditional_losses_70485232"
 dense_50/StatefulPartitionedCallÃ
.batch_normalization_61/StatefulPartitionedCallStatefulPartitionedCall)dense_50/StatefulPartitionedCall:output:0batch_normalization_61_7048788batch_normalization_61_7048790batch_normalization_61_7048792batch_normalization_61_7048794*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_61_layer_call_and_return_conditional_losses_704792220
.batch_normalization_61/StatefulPartitionedCall
ReluRelu7batch_normalization_61/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
Relu
 dense_51/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0dense_51_7048798*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_51_layer_call_and_return_conditional_losses_70485442"
 dense_51/StatefulPartitionedCallÃ
.batch_normalization_62/StatefulPartitionedCallStatefulPartitionedCall)dense_51/StatefulPartitionedCall:output:0batch_normalization_62_7048801batch_normalization_62_7048803batch_normalization_62_7048805batch_normalization_62_7048807*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_62_layer_call_and_return_conditional_losses_704808820
.batch_normalization_62/StatefulPartitionedCall
Relu_1Relu7batch_normalization_62/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
Relu_1
 dense_52/StatefulPartitionedCallStatefulPartitionedCallRelu_1:activations:0dense_52_7048811*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_52_layer_call_and_return_conditional_losses_70485652"
 dense_52/StatefulPartitionedCallÃ
.batch_normalization_63/StatefulPartitionedCallStatefulPartitionedCall)dense_52/StatefulPartitionedCall:output:0batch_normalization_63_7048814batch_normalization_63_7048816batch_normalization_63_7048818batch_normalization_63_7048820*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_63_layer_call_and_return_conditional_losses_704825420
.batch_normalization_63/StatefulPartitionedCall
Relu_2Relu7batch_normalization_63/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
Relu_2
 dense_53/StatefulPartitionedCallStatefulPartitionedCallRelu_2:activations:0dense_53_7048824*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_53_layer_call_and_return_conditional_losses_70485862"
 dense_53/StatefulPartitionedCallÃ
.batch_normalization_64/StatefulPartitionedCallStatefulPartitionedCall)dense_53/StatefulPartitionedCall:output:0batch_normalization_64_7048827batch_normalization_64_7048829batch_normalization_64_7048831batch_normalization_64_7048833*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_64_layer_call_and_return_conditional_losses_704842020
.batch_normalization_64/StatefulPartitionedCall
Relu_3Relu7batch_normalization_64/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
Relu_3¦
 dense_54/StatefulPartitionedCallStatefulPartitionedCallRelu_3:activations:0dense_54_7048837dense_54_7048839*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_54_layer_call_and_return_conditional_losses_70486102"
 dense_54/StatefulPartitionedCall
IdentityIdentity)dense_54/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identityò
NoOpNoOp/^batch_normalization_60/StatefulPartitionedCall/^batch_normalization_61/StatefulPartitionedCall/^batch_normalization_62/StatefulPartitionedCall/^batch_normalization_63/StatefulPartitionedCall/^batch_normalization_64/StatefulPartitionedCall!^dense_50/StatefulPartitionedCall!^dense_51/StatefulPartitionedCall!^dense_52/StatefulPartitionedCall!^dense_53/StatefulPartitionedCall!^dense_54/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:ÿÿÿÿÿÿÿÿÿÈ: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_60/StatefulPartitionedCall.batch_normalization_60/StatefulPartitionedCall2`
.batch_normalization_61/StatefulPartitionedCall.batch_normalization_61/StatefulPartitionedCall2`
.batch_normalization_62/StatefulPartitionedCall.batch_normalization_62/StatefulPartitionedCall2`
.batch_normalization_63/StatefulPartitionedCall.batch_normalization_63/StatefulPartitionedCall2`
.batch_normalization_64/StatefulPartitionedCall.batch_normalization_64/StatefulPartitionedCall2D
 dense_50/StatefulPartitionedCall dense_50/StatefulPartitionedCall2D
 dense_51/StatefulPartitionedCall dense_51/StatefulPartitionedCall2D
 dense_52/StatefulPartitionedCall dense_52/StatefulPartitionedCall2D
 dense_53/StatefulPartitionedCall dense_53/StatefulPartitionedCall2D
 dense_54/StatefulPartitionedCall dense_54/StatefulPartitionedCall:K G
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ

_user_specified_namex

°
E__inference_dense_50_layer_call_and_return_conditional_losses_7048523

inputs2
matmul_readvariableop_resource:
ÈÜ
identity¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ÈÜ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
MatMull
IdentityIdentityMatMul:product:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
ã
×
8__inference_batch_normalization_64_layer_call_fn_7050363

inputs
unknown:	Ü
	unknown_0:	Ü
	unknown_1:	Ü
	unknown_2:	Ü
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_64_layer_call_and_return_conditional_losses_70483582
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÜ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
 
_user_specified_nameinputs

¶
S__inference_batch_normalization_60_layer_call_and_return_conditional_losses_7049986

inputs0
!batchnorm_readvariableop_resource:	È4
%batchnorm_mul_readvariableop_resource:	È2
#batchnorm_readvariableop_1_resource:	È2
#batchnorm_readvariableop_2_resource:	È
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:È*
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:È2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:È2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:È*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:È2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:È*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:È2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:È*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:È2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

IdentityÂ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
ö,
ð
S__inference_batch_normalization_62_layer_call_and_return_conditional_losses_7050186

inputs6
'assignmovingavg_readvariableop_resource:	Ü8
)assignmovingavg_1_readvariableop_resource:	Ü4
%batchnorm_mul_readvariableop_resource:	Ü0
!batchnorm_readvariableop_resource:	Ü
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	Ü*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	Ü2
moments/StopGradient¥
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices³
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	Ü*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:Ü*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:Ü*
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
AssignMovingAvg/Cast¥
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:Ü*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:Ü2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes	
:Ü2
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
AssignMovingAvg_1/Cast«
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:Ü*
dtype02"
 AssignMovingAvg_1/ReadVariableOp¡
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:Ü2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes	
:Ü2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1k
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Ü2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ü2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2

Identityò
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÜ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
 
_user_specified_nameinputs
ö,
ð
S__inference_batch_normalization_64_layer_call_and_return_conditional_losses_7050350

inputs6
'assignmovingavg_readvariableop_resource:	Ü8
)assignmovingavg_1_readvariableop_resource:	Ü4
%batchnorm_mul_readvariableop_resource:	Ü0
!batchnorm_readvariableop_resource:	Ü
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	Ü*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	Ü2
moments/StopGradient¥
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices³
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	Ü*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:Ü*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:Ü*
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
AssignMovingAvg/Cast¥
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:Ü*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:Ü2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes	
:Ü2
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
AssignMovingAvg_1/Cast«
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:Ü*
dtype02"
 AssignMovingAvg_1/ReadVariableOp¡
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:Ü2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes	
:Ü2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1k
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Ü2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ü2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2

Identityò
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÜ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
 
_user_specified_nameinputs
ö,
ð
S__inference_batch_normalization_63_layer_call_and_return_conditional_losses_7050268

inputs6
'assignmovingavg_readvariableop_resource:	Ü8
)assignmovingavg_1_readvariableop_resource:	Ü4
%batchnorm_mul_readvariableop_resource:	Ü0
!batchnorm_readvariableop_resource:	Ü
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	Ü*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	Ü2
moments/StopGradient¥
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices³
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	Ü*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:Ü*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:Ü*
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
AssignMovingAvg/Cast¥
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:Ü*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:Ü2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes	
:Ü2
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
AssignMovingAvg_1/Cast«
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:Ü*
dtype02"
 AssignMovingAvg_1/ReadVariableOp¡
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:Ü2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes	
:Ü2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1k
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Ü2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ü2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:Ü*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ü2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2

Identityò
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÜ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
 
_user_specified_nameinputs
Ñ

*__inference_dense_51_layer_call_fn_7050404

inputs
unknown:
ÜÜ
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_51_layer_call_and_return_conditional_losses_70485442
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:ÿÿÿÿÿÿÿÿÿÜ: 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
 
_user_specified_nameinputs

°
E__inference_dense_52_layer_call_and_return_conditional_losses_7048565

inputs2
matmul_readvariableop_resource:
ÜÜ
identity¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ÜÜ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
MatMull
IdentityIdentityMatMul:product:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:ÿÿÿÿÿÿÿÿÿÜ: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
 
_user_specified_nameinputs
á
×
8__inference_batch_normalization_61_layer_call_fn_7050130

inputs
unknown:	Ü
	unknown_0:	Ü
	unknown_1:	Ü
	unknown_2:	Ü
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_61_layer_call_and_return_conditional_losses_70479222
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÜ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
 
_user_specified_nameinputs
èú
¿"
"__inference__wrapped_model_7047670
input_1_
Pfeed_forward_sub_net_10_batch_normalization_60_batchnorm_readvariableop_resource:	Èc
Tfeed_forward_sub_net_10_batch_normalization_60_batchnorm_mul_readvariableop_resource:	Èa
Rfeed_forward_sub_net_10_batch_normalization_60_batchnorm_readvariableop_1_resource:	Èa
Rfeed_forward_sub_net_10_batch_normalization_60_batchnorm_readvariableop_2_resource:	ÈS
?feed_forward_sub_net_10_dense_50_matmul_readvariableop_resource:
ÈÜ_
Pfeed_forward_sub_net_10_batch_normalization_61_batchnorm_readvariableop_resource:	Üc
Tfeed_forward_sub_net_10_batch_normalization_61_batchnorm_mul_readvariableop_resource:	Üa
Rfeed_forward_sub_net_10_batch_normalization_61_batchnorm_readvariableop_1_resource:	Üa
Rfeed_forward_sub_net_10_batch_normalization_61_batchnorm_readvariableop_2_resource:	ÜS
?feed_forward_sub_net_10_dense_51_matmul_readvariableop_resource:
ÜÜ_
Pfeed_forward_sub_net_10_batch_normalization_62_batchnorm_readvariableop_resource:	Üc
Tfeed_forward_sub_net_10_batch_normalization_62_batchnorm_mul_readvariableop_resource:	Üa
Rfeed_forward_sub_net_10_batch_normalization_62_batchnorm_readvariableop_1_resource:	Üa
Rfeed_forward_sub_net_10_batch_normalization_62_batchnorm_readvariableop_2_resource:	ÜS
?feed_forward_sub_net_10_dense_52_matmul_readvariableop_resource:
ÜÜ_
Pfeed_forward_sub_net_10_batch_normalization_63_batchnorm_readvariableop_resource:	Üc
Tfeed_forward_sub_net_10_batch_normalization_63_batchnorm_mul_readvariableop_resource:	Üa
Rfeed_forward_sub_net_10_batch_normalization_63_batchnorm_readvariableop_1_resource:	Üa
Rfeed_forward_sub_net_10_batch_normalization_63_batchnorm_readvariableop_2_resource:	ÜS
?feed_forward_sub_net_10_dense_53_matmul_readvariableop_resource:
ÜÜ_
Pfeed_forward_sub_net_10_batch_normalization_64_batchnorm_readvariableop_resource:	Üc
Tfeed_forward_sub_net_10_batch_normalization_64_batchnorm_mul_readvariableop_resource:	Üa
Rfeed_forward_sub_net_10_batch_normalization_64_batchnorm_readvariableop_1_resource:	Üa
Rfeed_forward_sub_net_10_batch_normalization_64_batchnorm_readvariableop_2_resource:	ÜS
?feed_forward_sub_net_10_dense_54_matmul_readvariableop_resource:
ÜÈO
@feed_forward_sub_net_10_dense_54_biasadd_readvariableop_resource:	È
identity¢Gfeed_forward_sub_net_10/batch_normalization_60/batchnorm/ReadVariableOp¢Ifeed_forward_sub_net_10/batch_normalization_60/batchnorm/ReadVariableOp_1¢Ifeed_forward_sub_net_10/batch_normalization_60/batchnorm/ReadVariableOp_2¢Kfeed_forward_sub_net_10/batch_normalization_60/batchnorm/mul/ReadVariableOp¢Gfeed_forward_sub_net_10/batch_normalization_61/batchnorm/ReadVariableOp¢Ifeed_forward_sub_net_10/batch_normalization_61/batchnorm/ReadVariableOp_1¢Ifeed_forward_sub_net_10/batch_normalization_61/batchnorm/ReadVariableOp_2¢Kfeed_forward_sub_net_10/batch_normalization_61/batchnorm/mul/ReadVariableOp¢Gfeed_forward_sub_net_10/batch_normalization_62/batchnorm/ReadVariableOp¢Ifeed_forward_sub_net_10/batch_normalization_62/batchnorm/ReadVariableOp_1¢Ifeed_forward_sub_net_10/batch_normalization_62/batchnorm/ReadVariableOp_2¢Kfeed_forward_sub_net_10/batch_normalization_62/batchnorm/mul/ReadVariableOp¢Gfeed_forward_sub_net_10/batch_normalization_63/batchnorm/ReadVariableOp¢Ifeed_forward_sub_net_10/batch_normalization_63/batchnorm/ReadVariableOp_1¢Ifeed_forward_sub_net_10/batch_normalization_63/batchnorm/ReadVariableOp_2¢Kfeed_forward_sub_net_10/batch_normalization_63/batchnorm/mul/ReadVariableOp¢Gfeed_forward_sub_net_10/batch_normalization_64/batchnorm/ReadVariableOp¢Ifeed_forward_sub_net_10/batch_normalization_64/batchnorm/ReadVariableOp_1¢Ifeed_forward_sub_net_10/batch_normalization_64/batchnorm/ReadVariableOp_2¢Kfeed_forward_sub_net_10/batch_normalization_64/batchnorm/mul/ReadVariableOp¢6feed_forward_sub_net_10/dense_50/MatMul/ReadVariableOp¢6feed_forward_sub_net_10/dense_51/MatMul/ReadVariableOp¢6feed_forward_sub_net_10/dense_52/MatMul/ReadVariableOp¢6feed_forward_sub_net_10/dense_53/MatMul/ReadVariableOp¢7feed_forward_sub_net_10/dense_54/BiasAdd/ReadVariableOp¢6feed_forward_sub_net_10/dense_54/MatMul/ReadVariableOp 
Gfeed_forward_sub_net_10/batch_normalization_60/batchnorm/ReadVariableOpReadVariableOpPfeed_forward_sub_net_10_batch_normalization_60_batchnorm_readvariableop_resource*
_output_shapes	
:È*
dtype02I
Gfeed_forward_sub_net_10/batch_normalization_60/batchnorm/ReadVariableOpÉ
>feed_forward_sub_net_10/batch_normalization_60/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2@
>feed_forward_sub_net_10/batch_normalization_60/batchnorm/add/yÅ
<feed_forward_sub_net_10/batch_normalization_60/batchnorm/addAddV2Ofeed_forward_sub_net_10/batch_normalization_60/batchnorm/ReadVariableOp:value:0Gfeed_forward_sub_net_10/batch_normalization_60/batchnorm/add/y:output:0*
T0*
_output_shapes	
:È2>
<feed_forward_sub_net_10/batch_normalization_60/batchnorm/addñ
>feed_forward_sub_net_10/batch_normalization_60/batchnorm/RsqrtRsqrt@feed_forward_sub_net_10/batch_normalization_60/batchnorm/add:z:0*
T0*
_output_shapes	
:È2@
>feed_forward_sub_net_10/batch_normalization_60/batchnorm/Rsqrt¬
Kfeed_forward_sub_net_10/batch_normalization_60/batchnorm/mul/ReadVariableOpReadVariableOpTfeed_forward_sub_net_10_batch_normalization_60_batchnorm_mul_readvariableop_resource*
_output_shapes	
:È*
dtype02M
Kfeed_forward_sub_net_10/batch_normalization_60/batchnorm/mul/ReadVariableOpÂ
<feed_forward_sub_net_10/batch_normalization_60/batchnorm/mulMulBfeed_forward_sub_net_10/batch_normalization_60/batchnorm/Rsqrt:y:0Sfeed_forward_sub_net_10/batch_normalization_60/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:È2>
<feed_forward_sub_net_10/batch_normalization_60/batchnorm/mul
>feed_forward_sub_net_10/batch_normalization_60/batchnorm/mul_1Mulinput_1@feed_forward_sub_net_10/batch_normalization_60/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2@
>feed_forward_sub_net_10/batch_normalization_60/batchnorm/mul_1¦
Ifeed_forward_sub_net_10/batch_normalization_60/batchnorm/ReadVariableOp_1ReadVariableOpRfeed_forward_sub_net_10_batch_normalization_60_batchnorm_readvariableop_1_resource*
_output_shapes	
:È*
dtype02K
Ifeed_forward_sub_net_10/batch_normalization_60/batchnorm/ReadVariableOp_1Â
>feed_forward_sub_net_10/batch_normalization_60/batchnorm/mul_2MulQfeed_forward_sub_net_10/batch_normalization_60/batchnorm/ReadVariableOp_1:value:0@feed_forward_sub_net_10/batch_normalization_60/batchnorm/mul:z:0*
T0*
_output_shapes	
:È2@
>feed_forward_sub_net_10/batch_normalization_60/batchnorm/mul_2¦
Ifeed_forward_sub_net_10/batch_normalization_60/batchnorm/ReadVariableOp_2ReadVariableOpRfeed_forward_sub_net_10_batch_normalization_60_batchnorm_readvariableop_2_resource*
_output_shapes	
:È*
dtype02K
Ifeed_forward_sub_net_10/batch_normalization_60/batchnorm/ReadVariableOp_2À
<feed_forward_sub_net_10/batch_normalization_60/batchnorm/subSubQfeed_forward_sub_net_10/batch_normalization_60/batchnorm/ReadVariableOp_2:value:0Bfeed_forward_sub_net_10/batch_normalization_60/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:È2>
<feed_forward_sub_net_10/batch_normalization_60/batchnorm/subÂ
>feed_forward_sub_net_10/batch_normalization_60/batchnorm/add_1AddV2Bfeed_forward_sub_net_10/batch_normalization_60/batchnorm/mul_1:z:0@feed_forward_sub_net_10/batch_normalization_60/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2@
>feed_forward_sub_net_10/batch_normalization_60/batchnorm/add_1ò
6feed_forward_sub_net_10/dense_50/MatMul/ReadVariableOpReadVariableOp?feed_forward_sub_net_10_dense_50_matmul_readvariableop_resource* 
_output_shapes
:
ÈÜ*
dtype028
6feed_forward_sub_net_10/dense_50/MatMul/ReadVariableOp
'feed_forward_sub_net_10/dense_50/MatMulMatMulBfeed_forward_sub_net_10/batch_normalization_60/batchnorm/add_1:z:0>feed_forward_sub_net_10/dense_50/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2)
'feed_forward_sub_net_10/dense_50/MatMul 
Gfeed_forward_sub_net_10/batch_normalization_61/batchnorm/ReadVariableOpReadVariableOpPfeed_forward_sub_net_10_batch_normalization_61_batchnorm_readvariableop_resource*
_output_shapes	
:Ü*
dtype02I
Gfeed_forward_sub_net_10/batch_normalization_61/batchnorm/ReadVariableOpÉ
>feed_forward_sub_net_10/batch_normalization_61/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2@
>feed_forward_sub_net_10/batch_normalization_61/batchnorm/add/yÅ
<feed_forward_sub_net_10/batch_normalization_61/batchnorm/addAddV2Ofeed_forward_sub_net_10/batch_normalization_61/batchnorm/ReadVariableOp:value:0Gfeed_forward_sub_net_10/batch_normalization_61/batchnorm/add/y:output:0*
T0*
_output_shapes	
:Ü2>
<feed_forward_sub_net_10/batch_normalization_61/batchnorm/addñ
>feed_forward_sub_net_10/batch_normalization_61/batchnorm/RsqrtRsqrt@feed_forward_sub_net_10/batch_normalization_61/batchnorm/add:z:0*
T0*
_output_shapes	
:Ü2@
>feed_forward_sub_net_10/batch_normalization_61/batchnorm/Rsqrt¬
Kfeed_forward_sub_net_10/batch_normalization_61/batchnorm/mul/ReadVariableOpReadVariableOpTfeed_forward_sub_net_10_batch_normalization_61_batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ü*
dtype02M
Kfeed_forward_sub_net_10/batch_normalization_61/batchnorm/mul/ReadVariableOpÂ
<feed_forward_sub_net_10/batch_normalization_61/batchnorm/mulMulBfeed_forward_sub_net_10/batch_normalization_61/batchnorm/Rsqrt:y:0Sfeed_forward_sub_net_10/batch_normalization_61/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ü2>
<feed_forward_sub_net_10/batch_normalization_61/batchnorm/mul¯
>feed_forward_sub_net_10/batch_normalization_61/batchnorm/mul_1Mul1feed_forward_sub_net_10/dense_50/MatMul:product:0@feed_forward_sub_net_10/batch_normalization_61/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2@
>feed_forward_sub_net_10/batch_normalization_61/batchnorm/mul_1¦
Ifeed_forward_sub_net_10/batch_normalization_61/batchnorm/ReadVariableOp_1ReadVariableOpRfeed_forward_sub_net_10_batch_normalization_61_batchnorm_readvariableop_1_resource*
_output_shapes	
:Ü*
dtype02K
Ifeed_forward_sub_net_10/batch_normalization_61/batchnorm/ReadVariableOp_1Â
>feed_forward_sub_net_10/batch_normalization_61/batchnorm/mul_2MulQfeed_forward_sub_net_10/batch_normalization_61/batchnorm/ReadVariableOp_1:value:0@feed_forward_sub_net_10/batch_normalization_61/batchnorm/mul:z:0*
T0*
_output_shapes	
:Ü2@
>feed_forward_sub_net_10/batch_normalization_61/batchnorm/mul_2¦
Ifeed_forward_sub_net_10/batch_normalization_61/batchnorm/ReadVariableOp_2ReadVariableOpRfeed_forward_sub_net_10_batch_normalization_61_batchnorm_readvariableop_2_resource*
_output_shapes	
:Ü*
dtype02K
Ifeed_forward_sub_net_10/batch_normalization_61/batchnorm/ReadVariableOp_2À
<feed_forward_sub_net_10/batch_normalization_61/batchnorm/subSubQfeed_forward_sub_net_10/batch_normalization_61/batchnorm/ReadVariableOp_2:value:0Bfeed_forward_sub_net_10/batch_normalization_61/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ü2>
<feed_forward_sub_net_10/batch_normalization_61/batchnorm/subÂ
>feed_forward_sub_net_10/batch_normalization_61/batchnorm/add_1AddV2Bfeed_forward_sub_net_10/batch_normalization_61/batchnorm/mul_1:z:0@feed_forward_sub_net_10/batch_normalization_61/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2@
>feed_forward_sub_net_10/batch_normalization_61/batchnorm/add_1»
feed_forward_sub_net_10/ReluReluBfeed_forward_sub_net_10/batch_normalization_61/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
feed_forward_sub_net_10/Reluò
6feed_forward_sub_net_10/dense_51/MatMul/ReadVariableOpReadVariableOp?feed_forward_sub_net_10_dense_51_matmul_readvariableop_resource* 
_output_shapes
:
ÜÜ*
dtype028
6feed_forward_sub_net_10/dense_51/MatMul/ReadVariableOpû
'feed_forward_sub_net_10/dense_51/MatMulMatMul*feed_forward_sub_net_10/Relu:activations:0>feed_forward_sub_net_10/dense_51/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2)
'feed_forward_sub_net_10/dense_51/MatMul 
Gfeed_forward_sub_net_10/batch_normalization_62/batchnorm/ReadVariableOpReadVariableOpPfeed_forward_sub_net_10_batch_normalization_62_batchnorm_readvariableop_resource*
_output_shapes	
:Ü*
dtype02I
Gfeed_forward_sub_net_10/batch_normalization_62/batchnorm/ReadVariableOpÉ
>feed_forward_sub_net_10/batch_normalization_62/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2@
>feed_forward_sub_net_10/batch_normalization_62/batchnorm/add/yÅ
<feed_forward_sub_net_10/batch_normalization_62/batchnorm/addAddV2Ofeed_forward_sub_net_10/batch_normalization_62/batchnorm/ReadVariableOp:value:0Gfeed_forward_sub_net_10/batch_normalization_62/batchnorm/add/y:output:0*
T0*
_output_shapes	
:Ü2>
<feed_forward_sub_net_10/batch_normalization_62/batchnorm/addñ
>feed_forward_sub_net_10/batch_normalization_62/batchnorm/RsqrtRsqrt@feed_forward_sub_net_10/batch_normalization_62/batchnorm/add:z:0*
T0*
_output_shapes	
:Ü2@
>feed_forward_sub_net_10/batch_normalization_62/batchnorm/Rsqrt¬
Kfeed_forward_sub_net_10/batch_normalization_62/batchnorm/mul/ReadVariableOpReadVariableOpTfeed_forward_sub_net_10_batch_normalization_62_batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ü*
dtype02M
Kfeed_forward_sub_net_10/batch_normalization_62/batchnorm/mul/ReadVariableOpÂ
<feed_forward_sub_net_10/batch_normalization_62/batchnorm/mulMulBfeed_forward_sub_net_10/batch_normalization_62/batchnorm/Rsqrt:y:0Sfeed_forward_sub_net_10/batch_normalization_62/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ü2>
<feed_forward_sub_net_10/batch_normalization_62/batchnorm/mul¯
>feed_forward_sub_net_10/batch_normalization_62/batchnorm/mul_1Mul1feed_forward_sub_net_10/dense_51/MatMul:product:0@feed_forward_sub_net_10/batch_normalization_62/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2@
>feed_forward_sub_net_10/batch_normalization_62/batchnorm/mul_1¦
Ifeed_forward_sub_net_10/batch_normalization_62/batchnorm/ReadVariableOp_1ReadVariableOpRfeed_forward_sub_net_10_batch_normalization_62_batchnorm_readvariableop_1_resource*
_output_shapes	
:Ü*
dtype02K
Ifeed_forward_sub_net_10/batch_normalization_62/batchnorm/ReadVariableOp_1Â
>feed_forward_sub_net_10/batch_normalization_62/batchnorm/mul_2MulQfeed_forward_sub_net_10/batch_normalization_62/batchnorm/ReadVariableOp_1:value:0@feed_forward_sub_net_10/batch_normalization_62/batchnorm/mul:z:0*
T0*
_output_shapes	
:Ü2@
>feed_forward_sub_net_10/batch_normalization_62/batchnorm/mul_2¦
Ifeed_forward_sub_net_10/batch_normalization_62/batchnorm/ReadVariableOp_2ReadVariableOpRfeed_forward_sub_net_10_batch_normalization_62_batchnorm_readvariableop_2_resource*
_output_shapes	
:Ü*
dtype02K
Ifeed_forward_sub_net_10/batch_normalization_62/batchnorm/ReadVariableOp_2À
<feed_forward_sub_net_10/batch_normalization_62/batchnorm/subSubQfeed_forward_sub_net_10/batch_normalization_62/batchnorm/ReadVariableOp_2:value:0Bfeed_forward_sub_net_10/batch_normalization_62/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ü2>
<feed_forward_sub_net_10/batch_normalization_62/batchnorm/subÂ
>feed_forward_sub_net_10/batch_normalization_62/batchnorm/add_1AddV2Bfeed_forward_sub_net_10/batch_normalization_62/batchnorm/mul_1:z:0@feed_forward_sub_net_10/batch_normalization_62/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2@
>feed_forward_sub_net_10/batch_normalization_62/batchnorm/add_1¿
feed_forward_sub_net_10/Relu_1ReluBfeed_forward_sub_net_10/batch_normalization_62/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2 
feed_forward_sub_net_10/Relu_1ò
6feed_forward_sub_net_10/dense_52/MatMul/ReadVariableOpReadVariableOp?feed_forward_sub_net_10_dense_52_matmul_readvariableop_resource* 
_output_shapes
:
ÜÜ*
dtype028
6feed_forward_sub_net_10/dense_52/MatMul/ReadVariableOpý
'feed_forward_sub_net_10/dense_52/MatMulMatMul,feed_forward_sub_net_10/Relu_1:activations:0>feed_forward_sub_net_10/dense_52/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2)
'feed_forward_sub_net_10/dense_52/MatMul 
Gfeed_forward_sub_net_10/batch_normalization_63/batchnorm/ReadVariableOpReadVariableOpPfeed_forward_sub_net_10_batch_normalization_63_batchnorm_readvariableop_resource*
_output_shapes	
:Ü*
dtype02I
Gfeed_forward_sub_net_10/batch_normalization_63/batchnorm/ReadVariableOpÉ
>feed_forward_sub_net_10/batch_normalization_63/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2@
>feed_forward_sub_net_10/batch_normalization_63/batchnorm/add/yÅ
<feed_forward_sub_net_10/batch_normalization_63/batchnorm/addAddV2Ofeed_forward_sub_net_10/batch_normalization_63/batchnorm/ReadVariableOp:value:0Gfeed_forward_sub_net_10/batch_normalization_63/batchnorm/add/y:output:0*
T0*
_output_shapes	
:Ü2>
<feed_forward_sub_net_10/batch_normalization_63/batchnorm/addñ
>feed_forward_sub_net_10/batch_normalization_63/batchnorm/RsqrtRsqrt@feed_forward_sub_net_10/batch_normalization_63/batchnorm/add:z:0*
T0*
_output_shapes	
:Ü2@
>feed_forward_sub_net_10/batch_normalization_63/batchnorm/Rsqrt¬
Kfeed_forward_sub_net_10/batch_normalization_63/batchnorm/mul/ReadVariableOpReadVariableOpTfeed_forward_sub_net_10_batch_normalization_63_batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ü*
dtype02M
Kfeed_forward_sub_net_10/batch_normalization_63/batchnorm/mul/ReadVariableOpÂ
<feed_forward_sub_net_10/batch_normalization_63/batchnorm/mulMulBfeed_forward_sub_net_10/batch_normalization_63/batchnorm/Rsqrt:y:0Sfeed_forward_sub_net_10/batch_normalization_63/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ü2>
<feed_forward_sub_net_10/batch_normalization_63/batchnorm/mul¯
>feed_forward_sub_net_10/batch_normalization_63/batchnorm/mul_1Mul1feed_forward_sub_net_10/dense_52/MatMul:product:0@feed_forward_sub_net_10/batch_normalization_63/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2@
>feed_forward_sub_net_10/batch_normalization_63/batchnorm/mul_1¦
Ifeed_forward_sub_net_10/batch_normalization_63/batchnorm/ReadVariableOp_1ReadVariableOpRfeed_forward_sub_net_10_batch_normalization_63_batchnorm_readvariableop_1_resource*
_output_shapes	
:Ü*
dtype02K
Ifeed_forward_sub_net_10/batch_normalization_63/batchnorm/ReadVariableOp_1Â
>feed_forward_sub_net_10/batch_normalization_63/batchnorm/mul_2MulQfeed_forward_sub_net_10/batch_normalization_63/batchnorm/ReadVariableOp_1:value:0@feed_forward_sub_net_10/batch_normalization_63/batchnorm/mul:z:0*
T0*
_output_shapes	
:Ü2@
>feed_forward_sub_net_10/batch_normalization_63/batchnorm/mul_2¦
Ifeed_forward_sub_net_10/batch_normalization_63/batchnorm/ReadVariableOp_2ReadVariableOpRfeed_forward_sub_net_10_batch_normalization_63_batchnorm_readvariableop_2_resource*
_output_shapes	
:Ü*
dtype02K
Ifeed_forward_sub_net_10/batch_normalization_63/batchnorm/ReadVariableOp_2À
<feed_forward_sub_net_10/batch_normalization_63/batchnorm/subSubQfeed_forward_sub_net_10/batch_normalization_63/batchnorm/ReadVariableOp_2:value:0Bfeed_forward_sub_net_10/batch_normalization_63/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ü2>
<feed_forward_sub_net_10/batch_normalization_63/batchnorm/subÂ
>feed_forward_sub_net_10/batch_normalization_63/batchnorm/add_1AddV2Bfeed_forward_sub_net_10/batch_normalization_63/batchnorm/mul_1:z:0@feed_forward_sub_net_10/batch_normalization_63/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2@
>feed_forward_sub_net_10/batch_normalization_63/batchnorm/add_1¿
feed_forward_sub_net_10/Relu_2ReluBfeed_forward_sub_net_10/batch_normalization_63/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2 
feed_forward_sub_net_10/Relu_2ò
6feed_forward_sub_net_10/dense_53/MatMul/ReadVariableOpReadVariableOp?feed_forward_sub_net_10_dense_53_matmul_readvariableop_resource* 
_output_shapes
:
ÜÜ*
dtype028
6feed_forward_sub_net_10/dense_53/MatMul/ReadVariableOpý
'feed_forward_sub_net_10/dense_53/MatMulMatMul,feed_forward_sub_net_10/Relu_2:activations:0>feed_forward_sub_net_10/dense_53/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2)
'feed_forward_sub_net_10/dense_53/MatMul 
Gfeed_forward_sub_net_10/batch_normalization_64/batchnorm/ReadVariableOpReadVariableOpPfeed_forward_sub_net_10_batch_normalization_64_batchnorm_readvariableop_resource*
_output_shapes	
:Ü*
dtype02I
Gfeed_forward_sub_net_10/batch_normalization_64/batchnorm/ReadVariableOpÉ
>feed_forward_sub_net_10/batch_normalization_64/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2@
>feed_forward_sub_net_10/batch_normalization_64/batchnorm/add/yÅ
<feed_forward_sub_net_10/batch_normalization_64/batchnorm/addAddV2Ofeed_forward_sub_net_10/batch_normalization_64/batchnorm/ReadVariableOp:value:0Gfeed_forward_sub_net_10/batch_normalization_64/batchnorm/add/y:output:0*
T0*
_output_shapes	
:Ü2>
<feed_forward_sub_net_10/batch_normalization_64/batchnorm/addñ
>feed_forward_sub_net_10/batch_normalization_64/batchnorm/RsqrtRsqrt@feed_forward_sub_net_10/batch_normalization_64/batchnorm/add:z:0*
T0*
_output_shapes	
:Ü2@
>feed_forward_sub_net_10/batch_normalization_64/batchnorm/Rsqrt¬
Kfeed_forward_sub_net_10/batch_normalization_64/batchnorm/mul/ReadVariableOpReadVariableOpTfeed_forward_sub_net_10_batch_normalization_64_batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ü*
dtype02M
Kfeed_forward_sub_net_10/batch_normalization_64/batchnorm/mul/ReadVariableOpÂ
<feed_forward_sub_net_10/batch_normalization_64/batchnorm/mulMulBfeed_forward_sub_net_10/batch_normalization_64/batchnorm/Rsqrt:y:0Sfeed_forward_sub_net_10/batch_normalization_64/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ü2>
<feed_forward_sub_net_10/batch_normalization_64/batchnorm/mul¯
>feed_forward_sub_net_10/batch_normalization_64/batchnorm/mul_1Mul1feed_forward_sub_net_10/dense_53/MatMul:product:0@feed_forward_sub_net_10/batch_normalization_64/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2@
>feed_forward_sub_net_10/batch_normalization_64/batchnorm/mul_1¦
Ifeed_forward_sub_net_10/batch_normalization_64/batchnorm/ReadVariableOp_1ReadVariableOpRfeed_forward_sub_net_10_batch_normalization_64_batchnorm_readvariableop_1_resource*
_output_shapes	
:Ü*
dtype02K
Ifeed_forward_sub_net_10/batch_normalization_64/batchnorm/ReadVariableOp_1Â
>feed_forward_sub_net_10/batch_normalization_64/batchnorm/mul_2MulQfeed_forward_sub_net_10/batch_normalization_64/batchnorm/ReadVariableOp_1:value:0@feed_forward_sub_net_10/batch_normalization_64/batchnorm/mul:z:0*
T0*
_output_shapes	
:Ü2@
>feed_forward_sub_net_10/batch_normalization_64/batchnorm/mul_2¦
Ifeed_forward_sub_net_10/batch_normalization_64/batchnorm/ReadVariableOp_2ReadVariableOpRfeed_forward_sub_net_10_batch_normalization_64_batchnorm_readvariableop_2_resource*
_output_shapes	
:Ü*
dtype02K
Ifeed_forward_sub_net_10/batch_normalization_64/batchnorm/ReadVariableOp_2À
<feed_forward_sub_net_10/batch_normalization_64/batchnorm/subSubQfeed_forward_sub_net_10/batch_normalization_64/batchnorm/ReadVariableOp_2:value:0Bfeed_forward_sub_net_10/batch_normalization_64/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ü2>
<feed_forward_sub_net_10/batch_normalization_64/batchnorm/subÂ
>feed_forward_sub_net_10/batch_normalization_64/batchnorm/add_1AddV2Bfeed_forward_sub_net_10/batch_normalization_64/batchnorm/mul_1:z:0@feed_forward_sub_net_10/batch_normalization_64/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2@
>feed_forward_sub_net_10/batch_normalization_64/batchnorm/add_1¿
feed_forward_sub_net_10/Relu_3ReluBfeed_forward_sub_net_10/batch_normalization_64/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2 
feed_forward_sub_net_10/Relu_3ò
6feed_forward_sub_net_10/dense_54/MatMul/ReadVariableOpReadVariableOp?feed_forward_sub_net_10_dense_54_matmul_readvariableop_resource* 
_output_shapes
:
ÜÈ*
dtype028
6feed_forward_sub_net_10/dense_54/MatMul/ReadVariableOpý
'feed_forward_sub_net_10/dense_54/MatMulMatMul,feed_forward_sub_net_10/Relu_3:activations:0>feed_forward_sub_net_10/dense_54/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2)
'feed_forward_sub_net_10/dense_54/MatMulð
7feed_forward_sub_net_10/dense_54/BiasAdd/ReadVariableOpReadVariableOp@feed_forward_sub_net_10_dense_54_biasadd_readvariableop_resource*
_output_shapes	
:È*
dtype029
7feed_forward_sub_net_10/dense_54/BiasAdd/ReadVariableOp
(feed_forward_sub_net_10/dense_54/BiasAddBiasAdd1feed_forward_sub_net_10/dense_54/MatMul:product:0?feed_forward_sub_net_10/dense_54/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2*
(feed_forward_sub_net_10/dense_54/BiasAdd
IdentityIdentity1feed_forward_sub_net_10/dense_54/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity
NoOpNoOpH^feed_forward_sub_net_10/batch_normalization_60/batchnorm/ReadVariableOpJ^feed_forward_sub_net_10/batch_normalization_60/batchnorm/ReadVariableOp_1J^feed_forward_sub_net_10/batch_normalization_60/batchnorm/ReadVariableOp_2L^feed_forward_sub_net_10/batch_normalization_60/batchnorm/mul/ReadVariableOpH^feed_forward_sub_net_10/batch_normalization_61/batchnorm/ReadVariableOpJ^feed_forward_sub_net_10/batch_normalization_61/batchnorm/ReadVariableOp_1J^feed_forward_sub_net_10/batch_normalization_61/batchnorm/ReadVariableOp_2L^feed_forward_sub_net_10/batch_normalization_61/batchnorm/mul/ReadVariableOpH^feed_forward_sub_net_10/batch_normalization_62/batchnorm/ReadVariableOpJ^feed_forward_sub_net_10/batch_normalization_62/batchnorm/ReadVariableOp_1J^feed_forward_sub_net_10/batch_normalization_62/batchnorm/ReadVariableOp_2L^feed_forward_sub_net_10/batch_normalization_62/batchnorm/mul/ReadVariableOpH^feed_forward_sub_net_10/batch_normalization_63/batchnorm/ReadVariableOpJ^feed_forward_sub_net_10/batch_normalization_63/batchnorm/ReadVariableOp_1J^feed_forward_sub_net_10/batch_normalization_63/batchnorm/ReadVariableOp_2L^feed_forward_sub_net_10/batch_normalization_63/batchnorm/mul/ReadVariableOpH^feed_forward_sub_net_10/batch_normalization_64/batchnorm/ReadVariableOpJ^feed_forward_sub_net_10/batch_normalization_64/batchnorm/ReadVariableOp_1J^feed_forward_sub_net_10/batch_normalization_64/batchnorm/ReadVariableOp_2L^feed_forward_sub_net_10/batch_normalization_64/batchnorm/mul/ReadVariableOp7^feed_forward_sub_net_10/dense_50/MatMul/ReadVariableOp7^feed_forward_sub_net_10/dense_51/MatMul/ReadVariableOp7^feed_forward_sub_net_10/dense_52/MatMul/ReadVariableOp7^feed_forward_sub_net_10/dense_53/MatMul/ReadVariableOp8^feed_forward_sub_net_10/dense_54/BiasAdd/ReadVariableOp7^feed_forward_sub_net_10/dense_54/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:ÿÿÿÿÿÿÿÿÿÈ: : : : : : : : : : : : : : : : : : : : : : : : : : 2
Gfeed_forward_sub_net_10/batch_normalization_60/batchnorm/ReadVariableOpGfeed_forward_sub_net_10/batch_normalization_60/batchnorm/ReadVariableOp2
Ifeed_forward_sub_net_10/batch_normalization_60/batchnorm/ReadVariableOp_1Ifeed_forward_sub_net_10/batch_normalization_60/batchnorm/ReadVariableOp_12
Ifeed_forward_sub_net_10/batch_normalization_60/batchnorm/ReadVariableOp_2Ifeed_forward_sub_net_10/batch_normalization_60/batchnorm/ReadVariableOp_22
Kfeed_forward_sub_net_10/batch_normalization_60/batchnorm/mul/ReadVariableOpKfeed_forward_sub_net_10/batch_normalization_60/batchnorm/mul/ReadVariableOp2
Gfeed_forward_sub_net_10/batch_normalization_61/batchnorm/ReadVariableOpGfeed_forward_sub_net_10/batch_normalization_61/batchnorm/ReadVariableOp2
Ifeed_forward_sub_net_10/batch_normalization_61/batchnorm/ReadVariableOp_1Ifeed_forward_sub_net_10/batch_normalization_61/batchnorm/ReadVariableOp_12
Ifeed_forward_sub_net_10/batch_normalization_61/batchnorm/ReadVariableOp_2Ifeed_forward_sub_net_10/batch_normalization_61/batchnorm/ReadVariableOp_22
Kfeed_forward_sub_net_10/batch_normalization_61/batchnorm/mul/ReadVariableOpKfeed_forward_sub_net_10/batch_normalization_61/batchnorm/mul/ReadVariableOp2
Gfeed_forward_sub_net_10/batch_normalization_62/batchnorm/ReadVariableOpGfeed_forward_sub_net_10/batch_normalization_62/batchnorm/ReadVariableOp2
Ifeed_forward_sub_net_10/batch_normalization_62/batchnorm/ReadVariableOp_1Ifeed_forward_sub_net_10/batch_normalization_62/batchnorm/ReadVariableOp_12
Ifeed_forward_sub_net_10/batch_normalization_62/batchnorm/ReadVariableOp_2Ifeed_forward_sub_net_10/batch_normalization_62/batchnorm/ReadVariableOp_22
Kfeed_forward_sub_net_10/batch_normalization_62/batchnorm/mul/ReadVariableOpKfeed_forward_sub_net_10/batch_normalization_62/batchnorm/mul/ReadVariableOp2
Gfeed_forward_sub_net_10/batch_normalization_63/batchnorm/ReadVariableOpGfeed_forward_sub_net_10/batch_normalization_63/batchnorm/ReadVariableOp2
Ifeed_forward_sub_net_10/batch_normalization_63/batchnorm/ReadVariableOp_1Ifeed_forward_sub_net_10/batch_normalization_63/batchnorm/ReadVariableOp_12
Ifeed_forward_sub_net_10/batch_normalization_63/batchnorm/ReadVariableOp_2Ifeed_forward_sub_net_10/batch_normalization_63/batchnorm/ReadVariableOp_22
Kfeed_forward_sub_net_10/batch_normalization_63/batchnorm/mul/ReadVariableOpKfeed_forward_sub_net_10/batch_normalization_63/batchnorm/mul/ReadVariableOp2
Gfeed_forward_sub_net_10/batch_normalization_64/batchnorm/ReadVariableOpGfeed_forward_sub_net_10/batch_normalization_64/batchnorm/ReadVariableOp2
Ifeed_forward_sub_net_10/batch_normalization_64/batchnorm/ReadVariableOp_1Ifeed_forward_sub_net_10/batch_normalization_64/batchnorm/ReadVariableOp_12
Ifeed_forward_sub_net_10/batch_normalization_64/batchnorm/ReadVariableOp_2Ifeed_forward_sub_net_10/batch_normalization_64/batchnorm/ReadVariableOp_22
Kfeed_forward_sub_net_10/batch_normalization_64/batchnorm/mul/ReadVariableOpKfeed_forward_sub_net_10/batch_normalization_64/batchnorm/mul/ReadVariableOp2p
6feed_forward_sub_net_10/dense_50/MatMul/ReadVariableOp6feed_forward_sub_net_10/dense_50/MatMul/ReadVariableOp2p
6feed_forward_sub_net_10/dense_51/MatMul/ReadVariableOp6feed_forward_sub_net_10/dense_51/MatMul/ReadVariableOp2p
6feed_forward_sub_net_10/dense_52/MatMul/ReadVariableOp6feed_forward_sub_net_10/dense_52/MatMul/ReadVariableOp2p
6feed_forward_sub_net_10/dense_53/MatMul/ReadVariableOp6feed_forward_sub_net_10/dense_53/MatMul/ReadVariableOp2r
7feed_forward_sub_net_10/dense_54/BiasAdd/ReadVariableOp7feed_forward_sub_net_10/dense_54/BiasAdd/ReadVariableOp2p
6feed_forward_sub_net_10/dense_54/MatMul/ReadVariableOp6feed_forward_sub_net_10/dense_54/MatMul/ReadVariableOp:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
!
_user_specified_name	input_1

°
E__inference_dense_53_layer_call_and_return_conditional_losses_7050425

inputs2
matmul_readvariableop_resource:
ÜÜ
identity¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ÜÜ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
MatMull
IdentityIdentityMatMul:product:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:ÿÿÿÿÿÿÿÿÿÜ: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
 
_user_specified_nameinputs

·
9__inference_feed_forward_sub_net_10_layer_call_fn_7049909
x
unknown:	È
	unknown_0:	È
	unknown_1:	È
	unknown_2:	È
	unknown_3:
ÈÜ
	unknown_4:	Ü
	unknown_5:	Ü
	unknown_6:	Ü
	unknown_7:	Ü
	unknown_8:
ÜÜ
	unknown_9:	Ü

unknown_10:	Ü

unknown_11:	Ü

unknown_12:	Ü

unknown_13:
ÜÜ

unknown_14:	Ü

unknown_15:	Ü

unknown_16:	Ü

unknown_17:	Ü

unknown_18:
ÜÜ

unknown_19:	Ü

unknown_20:	Ü

unknown_21:	Ü

unknown_22:	Ü

unknown_23:
ÜÈ

unknown_24:	È
identity¢StatefulPartitionedCall½
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
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_feed_forward_sub_net_10_layer_call_and_return_conditional_losses_70488432
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:ÿÿÿÿÿÿÿÿÿÈ: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ

_user_specified_namex
ã
×
8__inference_batch_normalization_61_layer_call_fn_7050117

inputs
unknown:	Ü
	unknown_0:	Ü
	unknown_1:	Ü
	unknown_2:	Ü
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_61_layer_call_and_return_conditional_losses_70478602
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÜ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
 
_user_specified_nameinputs
²

ù
E__inference_dense_54_layer_call_and_return_conditional_losses_7048610

inputs2
matmul_readvariableop_resource:
ÜÈ.
biasadd_readvariableop_resource:	È
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ÜÈ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:È*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2	
BiasAddl
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÜ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
 
_user_specified_nameinputs
²

T__inference_feed_forward_sub_net_10_layer_call_and_return_conditional_losses_7049552
input_1G
8batch_normalization_60_batchnorm_readvariableop_resource:	ÈK
<batch_normalization_60_batchnorm_mul_readvariableop_resource:	ÈI
:batch_normalization_60_batchnorm_readvariableop_1_resource:	ÈI
:batch_normalization_60_batchnorm_readvariableop_2_resource:	È;
'dense_50_matmul_readvariableop_resource:
ÈÜG
8batch_normalization_61_batchnorm_readvariableop_resource:	ÜK
<batch_normalization_61_batchnorm_mul_readvariableop_resource:	ÜI
:batch_normalization_61_batchnorm_readvariableop_1_resource:	ÜI
:batch_normalization_61_batchnorm_readvariableop_2_resource:	Ü;
'dense_51_matmul_readvariableop_resource:
ÜÜG
8batch_normalization_62_batchnorm_readvariableop_resource:	ÜK
<batch_normalization_62_batchnorm_mul_readvariableop_resource:	ÜI
:batch_normalization_62_batchnorm_readvariableop_1_resource:	ÜI
:batch_normalization_62_batchnorm_readvariableop_2_resource:	Ü;
'dense_52_matmul_readvariableop_resource:
ÜÜG
8batch_normalization_63_batchnorm_readvariableop_resource:	ÜK
<batch_normalization_63_batchnorm_mul_readvariableop_resource:	ÜI
:batch_normalization_63_batchnorm_readvariableop_1_resource:	ÜI
:batch_normalization_63_batchnorm_readvariableop_2_resource:	Ü;
'dense_53_matmul_readvariableop_resource:
ÜÜG
8batch_normalization_64_batchnorm_readvariableop_resource:	ÜK
<batch_normalization_64_batchnorm_mul_readvariableop_resource:	ÜI
:batch_normalization_64_batchnorm_readvariableop_1_resource:	ÜI
:batch_normalization_64_batchnorm_readvariableop_2_resource:	Ü;
'dense_54_matmul_readvariableop_resource:
ÜÈ7
(dense_54_biasadd_readvariableop_resource:	È
identity¢/batch_normalization_60/batchnorm/ReadVariableOp¢1batch_normalization_60/batchnorm/ReadVariableOp_1¢1batch_normalization_60/batchnorm/ReadVariableOp_2¢3batch_normalization_60/batchnorm/mul/ReadVariableOp¢/batch_normalization_61/batchnorm/ReadVariableOp¢1batch_normalization_61/batchnorm/ReadVariableOp_1¢1batch_normalization_61/batchnorm/ReadVariableOp_2¢3batch_normalization_61/batchnorm/mul/ReadVariableOp¢/batch_normalization_62/batchnorm/ReadVariableOp¢1batch_normalization_62/batchnorm/ReadVariableOp_1¢1batch_normalization_62/batchnorm/ReadVariableOp_2¢3batch_normalization_62/batchnorm/mul/ReadVariableOp¢/batch_normalization_63/batchnorm/ReadVariableOp¢1batch_normalization_63/batchnorm/ReadVariableOp_1¢1batch_normalization_63/batchnorm/ReadVariableOp_2¢3batch_normalization_63/batchnorm/mul/ReadVariableOp¢/batch_normalization_64/batchnorm/ReadVariableOp¢1batch_normalization_64/batchnorm/ReadVariableOp_1¢1batch_normalization_64/batchnorm/ReadVariableOp_2¢3batch_normalization_64/batchnorm/mul/ReadVariableOp¢dense_50/MatMul/ReadVariableOp¢dense_51/MatMul/ReadVariableOp¢dense_52/MatMul/ReadVariableOp¢dense_53/MatMul/ReadVariableOp¢dense_54/BiasAdd/ReadVariableOp¢dense_54/MatMul/ReadVariableOpØ
/batch_normalization_60/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_60_batchnorm_readvariableop_resource*
_output_shapes	
:È*
dtype021
/batch_normalization_60/batchnorm/ReadVariableOp
&batch_normalization_60/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_60/batchnorm/add/yå
$batch_normalization_60/batchnorm/addAddV27batch_normalization_60/batchnorm/ReadVariableOp:value:0/batch_normalization_60/batchnorm/add/y:output:0*
T0*
_output_shapes	
:È2&
$batch_normalization_60/batchnorm/add©
&batch_normalization_60/batchnorm/RsqrtRsqrt(batch_normalization_60/batchnorm/add:z:0*
T0*
_output_shapes	
:È2(
&batch_normalization_60/batchnorm/Rsqrtä
3batch_normalization_60/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_60_batchnorm_mul_readvariableop_resource*
_output_shapes	
:È*
dtype025
3batch_normalization_60/batchnorm/mul/ReadVariableOpâ
$batch_normalization_60/batchnorm/mulMul*batch_normalization_60/batchnorm/Rsqrt:y:0;batch_normalization_60/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:È2&
$batch_normalization_60/batchnorm/mul½
&batch_normalization_60/batchnorm/mul_1Mulinput_1(batch_normalization_60/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2(
&batch_normalization_60/batchnorm/mul_1Þ
1batch_normalization_60/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_60_batchnorm_readvariableop_1_resource*
_output_shapes	
:È*
dtype023
1batch_normalization_60/batchnorm/ReadVariableOp_1â
&batch_normalization_60/batchnorm/mul_2Mul9batch_normalization_60/batchnorm/ReadVariableOp_1:value:0(batch_normalization_60/batchnorm/mul:z:0*
T0*
_output_shapes	
:È2(
&batch_normalization_60/batchnorm/mul_2Þ
1batch_normalization_60/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_60_batchnorm_readvariableop_2_resource*
_output_shapes	
:È*
dtype023
1batch_normalization_60/batchnorm/ReadVariableOp_2à
$batch_normalization_60/batchnorm/subSub9batch_normalization_60/batchnorm/ReadVariableOp_2:value:0*batch_normalization_60/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:È2&
$batch_normalization_60/batchnorm/subâ
&batch_normalization_60/batchnorm/add_1AddV2*batch_normalization_60/batchnorm/mul_1:z:0(batch_normalization_60/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2(
&batch_normalization_60/batchnorm/add_1ª
dense_50/MatMul/ReadVariableOpReadVariableOp'dense_50_matmul_readvariableop_resource* 
_output_shapes
:
ÈÜ*
dtype02 
dense_50/MatMul/ReadVariableOp³
dense_50/MatMulMatMul*batch_normalization_60/batchnorm/add_1:z:0&dense_50/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
dense_50/MatMulØ
/batch_normalization_61/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_61_batchnorm_readvariableop_resource*
_output_shapes	
:Ü*
dtype021
/batch_normalization_61/batchnorm/ReadVariableOp
&batch_normalization_61/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_61/batchnorm/add/yå
$batch_normalization_61/batchnorm/addAddV27batch_normalization_61/batchnorm/ReadVariableOp:value:0/batch_normalization_61/batchnorm/add/y:output:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_61/batchnorm/add©
&batch_normalization_61/batchnorm/RsqrtRsqrt(batch_normalization_61/batchnorm/add:z:0*
T0*
_output_shapes	
:Ü2(
&batch_normalization_61/batchnorm/Rsqrtä
3batch_normalization_61/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_61_batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ü*
dtype025
3batch_normalization_61/batchnorm/mul/ReadVariableOpâ
$batch_normalization_61/batchnorm/mulMul*batch_normalization_61/batchnorm/Rsqrt:y:0;batch_normalization_61/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_61/batchnorm/mulÏ
&batch_normalization_61/batchnorm/mul_1Muldense_50/MatMul:product:0(batch_normalization_61/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2(
&batch_normalization_61/batchnorm/mul_1Þ
1batch_normalization_61/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_61_batchnorm_readvariableop_1_resource*
_output_shapes	
:Ü*
dtype023
1batch_normalization_61/batchnorm/ReadVariableOp_1â
&batch_normalization_61/batchnorm/mul_2Mul9batch_normalization_61/batchnorm/ReadVariableOp_1:value:0(batch_normalization_61/batchnorm/mul:z:0*
T0*
_output_shapes	
:Ü2(
&batch_normalization_61/batchnorm/mul_2Þ
1batch_normalization_61/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_61_batchnorm_readvariableop_2_resource*
_output_shapes	
:Ü*
dtype023
1batch_normalization_61/batchnorm/ReadVariableOp_2à
$batch_normalization_61/batchnorm/subSub9batch_normalization_61/batchnorm/ReadVariableOp_2:value:0*batch_normalization_61/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_61/batchnorm/subâ
&batch_normalization_61/batchnorm/add_1AddV2*batch_normalization_61/batchnorm/mul_1:z:0(batch_normalization_61/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2(
&batch_normalization_61/batchnorm/add_1s
ReluRelu*batch_normalization_61/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
Reluª
dense_51/MatMul/ReadVariableOpReadVariableOp'dense_51_matmul_readvariableop_resource* 
_output_shapes
:
ÜÜ*
dtype02 
dense_51/MatMul/ReadVariableOp
dense_51/MatMulMatMulRelu:activations:0&dense_51/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
dense_51/MatMulØ
/batch_normalization_62/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_62_batchnorm_readvariableop_resource*
_output_shapes	
:Ü*
dtype021
/batch_normalization_62/batchnorm/ReadVariableOp
&batch_normalization_62/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_62/batchnorm/add/yå
$batch_normalization_62/batchnorm/addAddV27batch_normalization_62/batchnorm/ReadVariableOp:value:0/batch_normalization_62/batchnorm/add/y:output:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_62/batchnorm/add©
&batch_normalization_62/batchnorm/RsqrtRsqrt(batch_normalization_62/batchnorm/add:z:0*
T0*
_output_shapes	
:Ü2(
&batch_normalization_62/batchnorm/Rsqrtä
3batch_normalization_62/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_62_batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ü*
dtype025
3batch_normalization_62/batchnorm/mul/ReadVariableOpâ
$batch_normalization_62/batchnorm/mulMul*batch_normalization_62/batchnorm/Rsqrt:y:0;batch_normalization_62/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_62/batchnorm/mulÏ
&batch_normalization_62/batchnorm/mul_1Muldense_51/MatMul:product:0(batch_normalization_62/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2(
&batch_normalization_62/batchnorm/mul_1Þ
1batch_normalization_62/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_62_batchnorm_readvariableop_1_resource*
_output_shapes	
:Ü*
dtype023
1batch_normalization_62/batchnorm/ReadVariableOp_1â
&batch_normalization_62/batchnorm/mul_2Mul9batch_normalization_62/batchnorm/ReadVariableOp_1:value:0(batch_normalization_62/batchnorm/mul:z:0*
T0*
_output_shapes	
:Ü2(
&batch_normalization_62/batchnorm/mul_2Þ
1batch_normalization_62/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_62_batchnorm_readvariableop_2_resource*
_output_shapes	
:Ü*
dtype023
1batch_normalization_62/batchnorm/ReadVariableOp_2à
$batch_normalization_62/batchnorm/subSub9batch_normalization_62/batchnorm/ReadVariableOp_2:value:0*batch_normalization_62/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_62/batchnorm/subâ
&batch_normalization_62/batchnorm/add_1AddV2*batch_normalization_62/batchnorm/mul_1:z:0(batch_normalization_62/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2(
&batch_normalization_62/batchnorm/add_1w
Relu_1Relu*batch_normalization_62/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
Relu_1ª
dense_52/MatMul/ReadVariableOpReadVariableOp'dense_52_matmul_readvariableop_resource* 
_output_shapes
:
ÜÜ*
dtype02 
dense_52/MatMul/ReadVariableOp
dense_52/MatMulMatMulRelu_1:activations:0&dense_52/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
dense_52/MatMulØ
/batch_normalization_63/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_63_batchnorm_readvariableop_resource*
_output_shapes	
:Ü*
dtype021
/batch_normalization_63/batchnorm/ReadVariableOp
&batch_normalization_63/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_63/batchnorm/add/yå
$batch_normalization_63/batchnorm/addAddV27batch_normalization_63/batchnorm/ReadVariableOp:value:0/batch_normalization_63/batchnorm/add/y:output:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_63/batchnorm/add©
&batch_normalization_63/batchnorm/RsqrtRsqrt(batch_normalization_63/batchnorm/add:z:0*
T0*
_output_shapes	
:Ü2(
&batch_normalization_63/batchnorm/Rsqrtä
3batch_normalization_63/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_63_batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ü*
dtype025
3batch_normalization_63/batchnorm/mul/ReadVariableOpâ
$batch_normalization_63/batchnorm/mulMul*batch_normalization_63/batchnorm/Rsqrt:y:0;batch_normalization_63/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_63/batchnorm/mulÏ
&batch_normalization_63/batchnorm/mul_1Muldense_52/MatMul:product:0(batch_normalization_63/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2(
&batch_normalization_63/batchnorm/mul_1Þ
1batch_normalization_63/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_63_batchnorm_readvariableop_1_resource*
_output_shapes	
:Ü*
dtype023
1batch_normalization_63/batchnorm/ReadVariableOp_1â
&batch_normalization_63/batchnorm/mul_2Mul9batch_normalization_63/batchnorm/ReadVariableOp_1:value:0(batch_normalization_63/batchnorm/mul:z:0*
T0*
_output_shapes	
:Ü2(
&batch_normalization_63/batchnorm/mul_2Þ
1batch_normalization_63/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_63_batchnorm_readvariableop_2_resource*
_output_shapes	
:Ü*
dtype023
1batch_normalization_63/batchnorm/ReadVariableOp_2à
$batch_normalization_63/batchnorm/subSub9batch_normalization_63/batchnorm/ReadVariableOp_2:value:0*batch_normalization_63/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_63/batchnorm/subâ
&batch_normalization_63/batchnorm/add_1AddV2*batch_normalization_63/batchnorm/mul_1:z:0(batch_normalization_63/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2(
&batch_normalization_63/batchnorm/add_1w
Relu_2Relu*batch_normalization_63/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
Relu_2ª
dense_53/MatMul/ReadVariableOpReadVariableOp'dense_53_matmul_readvariableop_resource* 
_output_shapes
:
ÜÜ*
dtype02 
dense_53/MatMul/ReadVariableOp
dense_53/MatMulMatMulRelu_2:activations:0&dense_53/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
dense_53/MatMulØ
/batch_normalization_64/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_64_batchnorm_readvariableop_resource*
_output_shapes	
:Ü*
dtype021
/batch_normalization_64/batchnorm/ReadVariableOp
&batch_normalization_64/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_64/batchnorm/add/yå
$batch_normalization_64/batchnorm/addAddV27batch_normalization_64/batchnorm/ReadVariableOp:value:0/batch_normalization_64/batchnorm/add/y:output:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_64/batchnorm/add©
&batch_normalization_64/batchnorm/RsqrtRsqrt(batch_normalization_64/batchnorm/add:z:0*
T0*
_output_shapes	
:Ü2(
&batch_normalization_64/batchnorm/Rsqrtä
3batch_normalization_64/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_64_batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ü*
dtype025
3batch_normalization_64/batchnorm/mul/ReadVariableOpâ
$batch_normalization_64/batchnorm/mulMul*batch_normalization_64/batchnorm/Rsqrt:y:0;batch_normalization_64/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_64/batchnorm/mulÏ
&batch_normalization_64/batchnorm/mul_1Muldense_53/MatMul:product:0(batch_normalization_64/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2(
&batch_normalization_64/batchnorm/mul_1Þ
1batch_normalization_64/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_64_batchnorm_readvariableop_1_resource*
_output_shapes	
:Ü*
dtype023
1batch_normalization_64/batchnorm/ReadVariableOp_1â
&batch_normalization_64/batchnorm/mul_2Mul9batch_normalization_64/batchnorm/ReadVariableOp_1:value:0(batch_normalization_64/batchnorm/mul:z:0*
T0*
_output_shapes	
:Ü2(
&batch_normalization_64/batchnorm/mul_2Þ
1batch_normalization_64/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_64_batchnorm_readvariableop_2_resource*
_output_shapes	
:Ü*
dtype023
1batch_normalization_64/batchnorm/ReadVariableOp_2à
$batch_normalization_64/batchnorm/subSub9batch_normalization_64/batchnorm/ReadVariableOp_2:value:0*batch_normalization_64/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ü2&
$batch_normalization_64/batchnorm/subâ
&batch_normalization_64/batchnorm/add_1AddV2*batch_normalization_64/batchnorm/mul_1:z:0(batch_normalization_64/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2(
&batch_normalization_64/batchnorm/add_1w
Relu_3Relu*batch_normalization_64/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
Relu_3ª
dense_54/MatMul/ReadVariableOpReadVariableOp'dense_54_matmul_readvariableop_resource* 
_output_shapes
:
ÜÈ*
dtype02 
dense_54/MatMul/ReadVariableOp
dense_54/MatMulMatMulRelu_3:activations:0&dense_54/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
dense_54/MatMul¨
dense_54/BiasAdd/ReadVariableOpReadVariableOp(dense_54_biasadd_readvariableop_resource*
_output_shapes	
:È*
dtype02!
dense_54/BiasAdd/ReadVariableOp¦
dense_54/BiasAddBiasAdddense_54/MatMul:product:0'dense_54/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
dense_54/BiasAddu
IdentityIdentitydense_54/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity¥

NoOpNoOp0^batch_normalization_60/batchnorm/ReadVariableOp2^batch_normalization_60/batchnorm/ReadVariableOp_12^batch_normalization_60/batchnorm/ReadVariableOp_24^batch_normalization_60/batchnorm/mul/ReadVariableOp0^batch_normalization_61/batchnorm/ReadVariableOp2^batch_normalization_61/batchnorm/ReadVariableOp_12^batch_normalization_61/batchnorm/ReadVariableOp_24^batch_normalization_61/batchnorm/mul/ReadVariableOp0^batch_normalization_62/batchnorm/ReadVariableOp2^batch_normalization_62/batchnorm/ReadVariableOp_12^batch_normalization_62/batchnorm/ReadVariableOp_24^batch_normalization_62/batchnorm/mul/ReadVariableOp0^batch_normalization_63/batchnorm/ReadVariableOp2^batch_normalization_63/batchnorm/ReadVariableOp_12^batch_normalization_63/batchnorm/ReadVariableOp_24^batch_normalization_63/batchnorm/mul/ReadVariableOp0^batch_normalization_64/batchnorm/ReadVariableOp2^batch_normalization_64/batchnorm/ReadVariableOp_12^batch_normalization_64/batchnorm/ReadVariableOp_24^batch_normalization_64/batchnorm/mul/ReadVariableOp^dense_50/MatMul/ReadVariableOp^dense_51/MatMul/ReadVariableOp^dense_52/MatMul/ReadVariableOp^dense_53/MatMul/ReadVariableOp ^dense_54/BiasAdd/ReadVariableOp^dense_54/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:ÿÿÿÿÿÿÿÿÿÈ: : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_60/batchnorm/ReadVariableOp/batch_normalization_60/batchnorm/ReadVariableOp2f
1batch_normalization_60/batchnorm/ReadVariableOp_11batch_normalization_60/batchnorm/ReadVariableOp_12f
1batch_normalization_60/batchnorm/ReadVariableOp_21batch_normalization_60/batchnorm/ReadVariableOp_22j
3batch_normalization_60/batchnorm/mul/ReadVariableOp3batch_normalization_60/batchnorm/mul/ReadVariableOp2b
/batch_normalization_61/batchnorm/ReadVariableOp/batch_normalization_61/batchnorm/ReadVariableOp2f
1batch_normalization_61/batchnorm/ReadVariableOp_11batch_normalization_61/batchnorm/ReadVariableOp_12f
1batch_normalization_61/batchnorm/ReadVariableOp_21batch_normalization_61/batchnorm/ReadVariableOp_22j
3batch_normalization_61/batchnorm/mul/ReadVariableOp3batch_normalization_61/batchnorm/mul/ReadVariableOp2b
/batch_normalization_62/batchnorm/ReadVariableOp/batch_normalization_62/batchnorm/ReadVariableOp2f
1batch_normalization_62/batchnorm/ReadVariableOp_11batch_normalization_62/batchnorm/ReadVariableOp_12f
1batch_normalization_62/batchnorm/ReadVariableOp_21batch_normalization_62/batchnorm/ReadVariableOp_22j
3batch_normalization_62/batchnorm/mul/ReadVariableOp3batch_normalization_62/batchnorm/mul/ReadVariableOp2b
/batch_normalization_63/batchnorm/ReadVariableOp/batch_normalization_63/batchnorm/ReadVariableOp2f
1batch_normalization_63/batchnorm/ReadVariableOp_11batch_normalization_63/batchnorm/ReadVariableOp_12f
1batch_normalization_63/batchnorm/ReadVariableOp_21batch_normalization_63/batchnorm/ReadVariableOp_22j
3batch_normalization_63/batchnorm/mul/ReadVariableOp3batch_normalization_63/batchnorm/mul/ReadVariableOp2b
/batch_normalization_64/batchnorm/ReadVariableOp/batch_normalization_64/batchnorm/ReadVariableOp2f
1batch_normalization_64/batchnorm/ReadVariableOp_11batch_normalization_64/batchnorm/ReadVariableOp_12f
1batch_normalization_64/batchnorm/ReadVariableOp_21batch_normalization_64/batchnorm/ReadVariableOp_22j
3batch_normalization_64/batchnorm/mul/ReadVariableOp3batch_normalization_64/batchnorm/mul/ReadVariableOp2@
dense_50/MatMul/ReadVariableOpdense_50/MatMul/ReadVariableOp2@
dense_51/MatMul/ReadVariableOpdense_51/MatMul/ReadVariableOp2@
dense_52/MatMul/ReadVariableOpdense_52/MatMul/ReadVariableOp2@
dense_53/MatMul/ReadVariableOpdense_53/MatMul/ReadVariableOp2B
dense_54/BiasAdd/ReadVariableOpdense_54/BiasAdd/ReadVariableOp2@
dense_54/MatMul/ReadVariableOpdense_54/MatMul/ReadVariableOp:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
!
_user_specified_name	input_1"¨L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*­
serving_default
<
input_11
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿÈ=
output_11
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿÈtensorflow/serving/predict:èË
ö
	bn_layers
dense_layers
trainable_variables
regularization_losses
	variables
	keras_api

signatures
+&call_and_return_all_conditional_losses
_default_save_signature
__call__"
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
10
11
12
 13
!14
"15"
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
#10
$11
%12
&13
'14
(15
)16
*17
+18
,19
20
21
22
 23
!24
"25"
trackable_list_wrapper
Î
trainable_variables

-layers
regularization_losses
.metrics
	variables
/non_trainable_variables
0layer_regularization_losses
1layer_metrics
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
ì
2axis
	gamma
beta
#moving_mean
$moving_variance
3trainable_variables
4regularization_losses
5	variables
6	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
ì
7axis
	gamma
beta
%moving_mean
&moving_variance
8trainable_variables
9regularization_losses
:	variables
;	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
ì
<axis
	gamma
beta
'moving_mean
(moving_variance
=trainable_variables
>regularization_losses
?	variables
@	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
ì
Aaxis
	gamma
beta
)moving_mean
*moving_variance
Btrainable_variables
Cregularization_losses
D	variables
E	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
ì
Faxis
	gamma
beta
+moving_mean
,moving_variance
Gtrainable_variables
Hregularization_losses
I	variables
J	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
(
K	keras_api"
_tf_keras_layer
³

kernel
Ltrainable_variables
Mregularization_losses
N	variables
O	keras_api
+ &call_and_return_all_conditional_losses
¡__call__"
_tf_keras_layer
³

kernel
Ptrainable_variables
Qregularization_losses
R	variables
S	keras_api
+¢&call_and_return_all_conditional_losses
£__call__"
_tf_keras_layer
³

kernel
Ttrainable_variables
Uregularization_losses
V	variables
W	keras_api
+¤&call_and_return_all_conditional_losses
¥__call__"
_tf_keras_layer
³

 kernel
Xtrainable_variables
Yregularization_losses
Z	variables
[	keras_api
+¦&call_and_return_all_conditional_losses
§__call__"
_tf_keras_layer
½

!kernel
"bias
\trainable_variables
]regularization_losses
^	variables
_	keras_api
+¨&call_and_return_all_conditional_losses
©__call__"
_tf_keras_layer
U:SÈ2Fnonshared_model_1/feed_forward_sub_net_10/batch_normalization_60/gamma
T:RÈ2Enonshared_model_1/feed_forward_sub_net_10/batch_normalization_60/beta
U:SÜ2Fnonshared_model_1/feed_forward_sub_net_10/batch_normalization_61/gamma
T:RÜ2Enonshared_model_1/feed_forward_sub_net_10/batch_normalization_61/beta
U:SÜ2Fnonshared_model_1/feed_forward_sub_net_10/batch_normalization_62/gamma
T:RÜ2Enonshared_model_1/feed_forward_sub_net_10/batch_normalization_62/beta
U:SÜ2Fnonshared_model_1/feed_forward_sub_net_10/batch_normalization_63/gamma
T:RÜ2Enonshared_model_1/feed_forward_sub_net_10/batch_normalization_63/beta
U:SÜ2Fnonshared_model_1/feed_forward_sub_net_10/batch_normalization_64/gamma
T:RÜ2Enonshared_model_1/feed_forward_sub_net_10/batch_normalization_64/beta
M:K
ÈÜ29nonshared_model_1/feed_forward_sub_net_10/dense_50/kernel
M:K
ÜÜ29nonshared_model_1/feed_forward_sub_net_10/dense_51/kernel
M:K
ÜÜ29nonshared_model_1/feed_forward_sub_net_10/dense_52/kernel
M:K
ÜÜ29nonshared_model_1/feed_forward_sub_net_10/dense_53/kernel
M:K
ÜÈ29nonshared_model_1/feed_forward_sub_net_10/dense_54/kernel
F:DÈ27nonshared_model_1/feed_forward_sub_net_10/dense_54/bias
]:[È (2Lnonshared_model_1/feed_forward_sub_net_10/batch_normalization_60/moving_mean
a:_È (2Pnonshared_model_1/feed_forward_sub_net_10/batch_normalization_60/moving_variance
]:[Ü (2Lnonshared_model_1/feed_forward_sub_net_10/batch_normalization_61/moving_mean
a:_Ü (2Pnonshared_model_1/feed_forward_sub_net_10/batch_normalization_61/moving_variance
]:[Ü (2Lnonshared_model_1/feed_forward_sub_net_10/batch_normalization_62/moving_mean
a:_Ü (2Pnonshared_model_1/feed_forward_sub_net_10/batch_normalization_62/moving_variance
]:[Ü (2Lnonshared_model_1/feed_forward_sub_net_10/batch_normalization_63/moving_mean
a:_Ü (2Pnonshared_model_1/feed_forward_sub_net_10/batch_normalization_63/moving_variance
]:[Ü (2Lnonshared_model_1/feed_forward_sub_net_10/batch_normalization_64/moving_mean
a:_Ü (2Pnonshared_model_1/feed_forward_sub_net_10/batch_normalization_64/moving_variance
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
trackable_list_wrapper
f
#0
$1
%2
&3
'4
(5
)6
*7
+8
,9"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
#2
$3"
trackable_list_wrapper
°
3trainable_variables

`layers
4regularization_losses
ametrics
5	variables
bnon_trainable_variables
clayer_regularization_losses
dlayer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
%2
&3"
trackable_list_wrapper
°
8trainable_variables

elayers
9regularization_losses
fmetrics
:	variables
gnon_trainable_variables
hlayer_regularization_losses
ilayer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
'2
(3"
trackable_list_wrapper
°
=trainable_variables

jlayers
>regularization_losses
kmetrics
?	variables
lnon_trainable_variables
mlayer_regularization_losses
nlayer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
)2
*3"
trackable_list_wrapper
°
Btrainable_variables

olayers
Cregularization_losses
pmetrics
D	variables
qnon_trainable_variables
rlayer_regularization_losses
slayer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
+2
,3"
trackable_list_wrapper
°
Gtrainable_variables

tlayers
Hregularization_losses
umetrics
I	variables
vnon_trainable_variables
wlayer_regularization_losses
xlayer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
°
Ltrainable_variables

ylayers
Mregularization_losses
zmetrics
N	variables
{non_trainable_variables
|layer_regularization_losses
}layer_metrics
¡__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
³
Ptrainable_variables

~layers
Qregularization_losses
metrics
R	variables
non_trainable_variables
 layer_regularization_losses
layer_metrics
£__call__
+¢&call_and_return_all_conditional_losses
'¢"call_and_return_conditional_losses"
_generic_user_object
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
µ
Ttrainable_variables
layers
Uregularization_losses
metrics
V	variables
non_trainable_variables
 layer_regularization_losses
layer_metrics
¥__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses"
_generic_user_object
'
 0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
 0"
trackable_list_wrapper
µ
Xtrainable_variables
layers
Yregularization_losses
metrics
Z	variables
non_trainable_variables
 layer_regularization_losses
layer_metrics
§__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses"
_generic_user_object
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
µ
\trainable_variables
layers
]regularization_losses
metrics
^	variables
non_trainable_variables
 layer_regularization_losses
layer_metrics
©__call__
+¨&call_and_return_all_conditional_losses
'¨"call_and_return_conditional_losses"
_generic_user_object
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
.
'0
(1"
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
)0
*1"
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
+0
,1"
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
2
T__inference_feed_forward_sub_net_10_layer_call_and_return_conditional_losses_7049260
T__inference_feed_forward_sub_net_10_layer_call_and_return_conditional_losses_7049446
T__inference_feed_forward_sub_net_10_layer_call_and_return_conditional_losses_7049552
T__inference_feed_forward_sub_net_10_layer_call_and_return_conditional_losses_7049738«
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
ÍBÊ
"__inference__wrapped_model_7047670input_1"
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
2
9__inference_feed_forward_sub_net_10_layer_call_fn_7049795
9__inference_feed_forward_sub_net_10_layer_call_fn_7049852
9__inference_feed_forward_sub_net_10_layer_call_fn_7049909
9__inference_feed_forward_sub_net_10_layer_call_fn_7049966«
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
ÌBÉ
%__inference_signature_wrapper_7049154input_1"
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
ä2á
S__inference_batch_normalization_60_layer_call_and_return_conditional_losses_7049986
S__inference_batch_normalization_60_layer_call_and_return_conditional_losses_7050022´
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
®2«
8__inference_batch_normalization_60_layer_call_fn_7050035
8__inference_batch_normalization_60_layer_call_fn_7050048´
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
ä2á
S__inference_batch_normalization_61_layer_call_and_return_conditional_losses_7050068
S__inference_batch_normalization_61_layer_call_and_return_conditional_losses_7050104´
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
®2«
8__inference_batch_normalization_61_layer_call_fn_7050117
8__inference_batch_normalization_61_layer_call_fn_7050130´
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
ä2á
S__inference_batch_normalization_62_layer_call_and_return_conditional_losses_7050150
S__inference_batch_normalization_62_layer_call_and_return_conditional_losses_7050186´
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
®2«
8__inference_batch_normalization_62_layer_call_fn_7050199
8__inference_batch_normalization_62_layer_call_fn_7050212´
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
ä2á
S__inference_batch_normalization_63_layer_call_and_return_conditional_losses_7050232
S__inference_batch_normalization_63_layer_call_and_return_conditional_losses_7050268´
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
®2«
8__inference_batch_normalization_63_layer_call_fn_7050281
8__inference_batch_normalization_63_layer_call_fn_7050294´
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
ä2á
S__inference_batch_normalization_64_layer_call_and_return_conditional_losses_7050314
S__inference_batch_normalization_64_layer_call_and_return_conditional_losses_7050350´
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
®2«
8__inference_batch_normalization_64_layer_call_fn_7050363
8__inference_batch_normalization_64_layer_call_fn_7050376´
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
ï2ì
E__inference_dense_50_layer_call_and_return_conditional_losses_7050383¢
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
Ô2Ñ
*__inference_dense_50_layer_call_fn_7050390¢
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
ï2ì
E__inference_dense_51_layer_call_and_return_conditional_losses_7050397¢
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
Ô2Ñ
*__inference_dense_51_layer_call_fn_7050404¢
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
ï2ì
E__inference_dense_52_layer_call_and_return_conditional_losses_7050411¢
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
Ô2Ñ
*__inference_dense_52_layer_call_fn_7050418¢
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
ï2ì
E__inference_dense_53_layer_call_and_return_conditional_losses_7050425¢
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
Ô2Ñ
*__inference_dense_53_layer_call_fn_7050432¢
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
ï2ì
E__inference_dense_54_layer_call_and_return_conditional_losses_7050442¢
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
Ô2Ñ
*__inference_dense_54_layer_call_fn_7050451¢
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
 ¬
"__inference__wrapped_model_7047670$#&%('*) ,+!"1¢.
'¢$
"
input_1ÿÿÿÿÿÿÿÿÿÈ
ª "4ª1
/
output_1# 
output_1ÿÿÿÿÿÿÿÿÿÈ»
S__inference_batch_normalization_60_layer_call_and_return_conditional_losses_7049986d$#4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿÈ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÈ
 »
S__inference_batch_normalization_60_layer_call_and_return_conditional_losses_7050022d#$4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿÈ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÈ
 
8__inference_batch_normalization_60_layer_call_fn_7050035W$#4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿÈ
p 
ª "ÿÿÿÿÿÿÿÿÿÈ
8__inference_batch_normalization_60_layer_call_fn_7050048W#$4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿÈ
p
ª "ÿÿÿÿÿÿÿÿÿÈ»
S__inference_batch_normalization_61_layer_call_and_return_conditional_losses_7050068d&%4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿÜ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÜ
 »
S__inference_batch_normalization_61_layer_call_and_return_conditional_losses_7050104d%&4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿÜ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÜ
 
8__inference_batch_normalization_61_layer_call_fn_7050117W&%4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿÜ
p 
ª "ÿÿÿÿÿÿÿÿÿÜ
8__inference_batch_normalization_61_layer_call_fn_7050130W%&4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿÜ
p
ª "ÿÿÿÿÿÿÿÿÿÜ»
S__inference_batch_normalization_62_layer_call_and_return_conditional_losses_7050150d('4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿÜ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÜ
 »
S__inference_batch_normalization_62_layer_call_and_return_conditional_losses_7050186d'(4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿÜ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÜ
 
8__inference_batch_normalization_62_layer_call_fn_7050199W('4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿÜ
p 
ª "ÿÿÿÿÿÿÿÿÿÜ
8__inference_batch_normalization_62_layer_call_fn_7050212W'(4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿÜ
p
ª "ÿÿÿÿÿÿÿÿÿÜ»
S__inference_batch_normalization_63_layer_call_and_return_conditional_losses_7050232d*)4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿÜ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÜ
 »
S__inference_batch_normalization_63_layer_call_and_return_conditional_losses_7050268d)*4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿÜ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÜ
 
8__inference_batch_normalization_63_layer_call_fn_7050281W*)4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿÜ
p 
ª "ÿÿÿÿÿÿÿÿÿÜ
8__inference_batch_normalization_63_layer_call_fn_7050294W)*4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿÜ
p
ª "ÿÿÿÿÿÿÿÿÿÜ»
S__inference_batch_normalization_64_layer_call_and_return_conditional_losses_7050314d,+4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿÜ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÜ
 »
S__inference_batch_normalization_64_layer_call_and_return_conditional_losses_7050350d+,4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿÜ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÜ
 
8__inference_batch_normalization_64_layer_call_fn_7050363W,+4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿÜ
p 
ª "ÿÿÿÿÿÿÿÿÿÜ
8__inference_batch_normalization_64_layer_call_fn_7050376W+,4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿÜ
p
ª "ÿÿÿÿÿÿÿÿÿÜ¦
E__inference_dense_50_layer_call_and_return_conditional_losses_7050383]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÈ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÜ
 ~
*__inference_dense_50_layer_call_fn_7050390P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÈ
ª "ÿÿÿÿÿÿÿÿÿÜ¦
E__inference_dense_51_layer_call_and_return_conditional_losses_7050397]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÜ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÜ
 ~
*__inference_dense_51_layer_call_fn_7050404P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÜ
ª "ÿÿÿÿÿÿÿÿÿÜ¦
E__inference_dense_52_layer_call_and_return_conditional_losses_7050411]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÜ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÜ
 ~
*__inference_dense_52_layer_call_fn_7050418P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÜ
ª "ÿÿÿÿÿÿÿÿÿÜ¦
E__inference_dense_53_layer_call_and_return_conditional_losses_7050425] 0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÜ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÜ
 ~
*__inference_dense_53_layer_call_fn_7050432P 0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÜ
ª "ÿÿÿÿÿÿÿÿÿÜ§
E__inference_dense_54_layer_call_and_return_conditional_losses_7050442^!"0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÜ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÈ
 
*__inference_dense_54_layer_call_fn_7050451Q!"0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÜ
ª "ÿÿÿÿÿÿÿÿÿÈÍ
T__inference_feed_forward_sub_net_10_layer_call_and_return_conditional_losses_7049260u$#&%('*) ,+!"/¢,
%¢"

xÿÿÿÿÿÿÿÿÿÈ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÈ
 Í
T__inference_feed_forward_sub_net_10_layer_call_and_return_conditional_losses_7049446u#$%&'()* +,!"/¢,
%¢"

xÿÿÿÿÿÿÿÿÿÈ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÈ
 Ó
T__inference_feed_forward_sub_net_10_layer_call_and_return_conditional_losses_7049552{$#&%('*) ,+!"5¢2
+¢(
"
input_1ÿÿÿÿÿÿÿÿÿÈ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÈ
 Ó
T__inference_feed_forward_sub_net_10_layer_call_and_return_conditional_losses_7049738{#$%&'()* +,!"5¢2
+¢(
"
input_1ÿÿÿÿÿÿÿÿÿÈ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÈ
 «
9__inference_feed_forward_sub_net_10_layer_call_fn_7049795n$#&%('*) ,+!"5¢2
+¢(
"
input_1ÿÿÿÿÿÿÿÿÿÈ
p 
ª "ÿÿÿÿÿÿÿÿÿÈ¥
9__inference_feed_forward_sub_net_10_layer_call_fn_7049852h$#&%('*) ,+!"/¢,
%¢"

xÿÿÿÿÿÿÿÿÿÈ
p 
ª "ÿÿÿÿÿÿÿÿÿÈ¥
9__inference_feed_forward_sub_net_10_layer_call_fn_7049909h#$%&'()* +,!"/¢,
%¢"

xÿÿÿÿÿÿÿÿÿÈ
p
ª "ÿÿÿÿÿÿÿÿÿÈ«
9__inference_feed_forward_sub_net_10_layer_call_fn_7049966n#$%&'()* +,!"5¢2
+¢(
"
input_1ÿÿÿÿÿÿÿÿÿÈ
p
ª "ÿÿÿÿÿÿÿÿÿÈº
%__inference_signature_wrapper_7049154$#&%('*) ,+!"<¢9
¢ 
2ª/
-
input_1"
input_1ÿÿÿÿÿÿÿÿÿÈ"4ª1
/
output_1# 
output_1ÿÿÿÿÿÿÿÿÿÈ