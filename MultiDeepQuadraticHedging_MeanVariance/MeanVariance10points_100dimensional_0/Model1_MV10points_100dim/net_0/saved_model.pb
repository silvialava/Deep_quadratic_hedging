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
 �"serve*2.6.02unknown8��
�
.feed_forward_sub_net/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*?
shared_name0.feed_forward_sub_net/batch_normalization/gamma
�
Bfeed_forward_sub_net/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp.feed_forward_sub_net/batch_normalization/gamma*
_output_shapes	
:�*
dtype0
�
-feed_forward_sub_net/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*>
shared_name/-feed_forward_sub_net/batch_normalization/beta
�
Afeed_forward_sub_net/batch_normalization/beta/Read/ReadVariableOpReadVariableOp-feed_forward_sub_net/batch_normalization/beta*
_output_shapes	
:�*
dtype0
�
0feed_forward_sub_net/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*A
shared_name20feed_forward_sub_net/batch_normalization_1/gamma
�
Dfeed_forward_sub_net/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp0feed_forward_sub_net/batch_normalization_1/gamma*
_output_shapes	
:�*
dtype0
�
/feed_forward_sub_net/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*@
shared_name1/feed_forward_sub_net/batch_normalization_1/beta
�
Cfeed_forward_sub_net/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp/feed_forward_sub_net/batch_normalization_1/beta*
_output_shapes	
:�*
dtype0
�
0feed_forward_sub_net/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*A
shared_name20feed_forward_sub_net/batch_normalization_2/gamma
�
Dfeed_forward_sub_net/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp0feed_forward_sub_net/batch_normalization_2/gamma*
_output_shapes	
:�*
dtype0
�
/feed_forward_sub_net/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*@
shared_name1/feed_forward_sub_net/batch_normalization_2/beta
�
Cfeed_forward_sub_net/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp/feed_forward_sub_net/batch_normalization_2/beta*
_output_shapes	
:�*
dtype0
�
0feed_forward_sub_net/batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*A
shared_name20feed_forward_sub_net/batch_normalization_3/gamma
�
Dfeed_forward_sub_net/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOp0feed_forward_sub_net/batch_normalization_3/gamma*
_output_shapes	
:�*
dtype0
�
/feed_forward_sub_net/batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*@
shared_name1/feed_forward_sub_net/batch_normalization_3/beta
�
Cfeed_forward_sub_net/batch_normalization_3/beta/Read/ReadVariableOpReadVariableOp/feed_forward_sub_net/batch_normalization_3/beta*
_output_shapes	
:�*
dtype0
�
0feed_forward_sub_net/batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*A
shared_name20feed_forward_sub_net/batch_normalization_4/gamma
�
Dfeed_forward_sub_net/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOp0feed_forward_sub_net/batch_normalization_4/gamma*
_output_shapes	
:�*
dtype0
�
/feed_forward_sub_net/batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*@
shared_name1/feed_forward_sub_net/batch_normalization_4/beta
�
Cfeed_forward_sub_net/batch_normalization_4/beta/Read/ReadVariableOpReadVariableOp/feed_forward_sub_net/batch_normalization_4/beta*
_output_shapes	
:�*
dtype0
�
!feed_forward_sub_net/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*2
shared_name#!feed_forward_sub_net/dense/kernel
�
5feed_forward_sub_net/dense/kernel/Read/ReadVariableOpReadVariableOp!feed_forward_sub_net/dense/kernel* 
_output_shapes
:
��*
dtype0
�
#feed_forward_sub_net/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*4
shared_name%#feed_forward_sub_net/dense_1/kernel
�
7feed_forward_sub_net/dense_1/kernel/Read/ReadVariableOpReadVariableOp#feed_forward_sub_net/dense_1/kernel* 
_output_shapes
:
��*
dtype0
�
#feed_forward_sub_net/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*4
shared_name%#feed_forward_sub_net/dense_2/kernel
�
7feed_forward_sub_net/dense_2/kernel/Read/ReadVariableOpReadVariableOp#feed_forward_sub_net/dense_2/kernel* 
_output_shapes
:
��*
dtype0
�
#feed_forward_sub_net/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*4
shared_name%#feed_forward_sub_net/dense_3/kernel
�
7feed_forward_sub_net/dense_3/kernel/Read/ReadVariableOpReadVariableOp#feed_forward_sub_net/dense_3/kernel* 
_output_shapes
:
��*
dtype0
�
#feed_forward_sub_net/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*4
shared_name%#feed_forward_sub_net/dense_4/kernel
�
7feed_forward_sub_net/dense_4/kernel/Read/ReadVariableOpReadVariableOp#feed_forward_sub_net/dense_4/kernel* 
_output_shapes
:
��*
dtype0
�
!feed_forward_sub_net/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!feed_forward_sub_net/dense_4/bias
�
5feed_forward_sub_net/dense_4/bias/Read/ReadVariableOpReadVariableOp!feed_forward_sub_net/dense_4/bias*
_output_shapes	
:�*
dtype0
�
4feed_forward_sub_net/batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*E
shared_name64feed_forward_sub_net/batch_normalization/moving_mean
�
Hfeed_forward_sub_net/batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOp4feed_forward_sub_net/batch_normalization/moving_mean*
_output_shapes	
:�*
dtype0
�
8feed_forward_sub_net/batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*I
shared_name:8feed_forward_sub_net/batch_normalization/moving_variance
�
Lfeed_forward_sub_net/batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp8feed_forward_sub_net/batch_normalization/moving_variance*
_output_shapes	
:�*
dtype0
�
6feed_forward_sub_net/batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*G
shared_name86feed_forward_sub_net/batch_normalization_1/moving_mean
�
Jfeed_forward_sub_net/batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp6feed_forward_sub_net/batch_normalization_1/moving_mean*
_output_shapes	
:�*
dtype0
�
:feed_forward_sub_net/batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*K
shared_name<:feed_forward_sub_net/batch_normalization_1/moving_variance
�
Nfeed_forward_sub_net/batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp:feed_forward_sub_net/batch_normalization_1/moving_variance*
_output_shapes	
:�*
dtype0
�
6feed_forward_sub_net/batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*G
shared_name86feed_forward_sub_net/batch_normalization_2/moving_mean
�
Jfeed_forward_sub_net/batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp6feed_forward_sub_net/batch_normalization_2/moving_mean*
_output_shapes	
:�*
dtype0
�
:feed_forward_sub_net/batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*K
shared_name<:feed_forward_sub_net/batch_normalization_2/moving_variance
�
Nfeed_forward_sub_net/batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp:feed_forward_sub_net/batch_normalization_2/moving_variance*
_output_shapes	
:�*
dtype0
�
6feed_forward_sub_net/batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*G
shared_name86feed_forward_sub_net/batch_normalization_3/moving_mean
�
Jfeed_forward_sub_net/batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp6feed_forward_sub_net/batch_normalization_3/moving_mean*
_output_shapes	
:�*
dtype0
�
:feed_forward_sub_net/batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*K
shared_name<:feed_forward_sub_net/batch_normalization_3/moving_variance
�
Nfeed_forward_sub_net/batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp:feed_forward_sub_net/batch_normalization_3/moving_variance*
_output_shapes	
:�*
dtype0
�
6feed_forward_sub_net/batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*G
shared_name86feed_forward_sub_net/batch_normalization_4/moving_mean
�
Jfeed_forward_sub_net/batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp6feed_forward_sub_net/batch_normalization_4/moving_mean*
_output_shapes	
:�*
dtype0
�
:feed_forward_sub_net/batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*K
shared_name<:feed_forward_sub_net/batch_normalization_4/moving_variance
�
Nfeed_forward_sub_net/batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp:feed_forward_sub_net/batch_normalization_4/moving_variance*
_output_shapes	
:�*
dtype0

NoOpNoOp
�;
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�:
value�:B�: B�:
�
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
�
trainable_variables

-layers
regularization_losses
.metrics
	variables
/non_trainable_variables
0layer_regularization_losses
1layer_metrics
 
�
2axis
	gamma
beta
#moving_mean
$moving_variance
3trainable_variables
4regularization_losses
5	variables
6	keras_api
�
7axis
	gamma
beta
%moving_mean
&moving_variance
8trainable_variables
9regularization_losses
:	variables
;	keras_api
�
<axis
	gamma
beta
'moving_mean
(moving_variance
=trainable_variables
>regularization_losses
?	variables
@	keras_api
�
Aaxis
	gamma
beta
)moving_mean
*moving_variance
Btrainable_variables
Cregularization_losses
D	variables
E	keras_api
�
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
tr
VARIABLE_VALUE.feed_forward_sub_net/batch_normalization/gamma0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE-feed_forward_sub_net/batch_normalization/beta0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE0feed_forward_sub_net/batch_normalization_1/gamma0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE/feed_forward_sub_net/batch_normalization_1/beta0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE0feed_forward_sub_net/batch_normalization_2/gamma0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE/feed_forward_sub_net/batch_normalization_2/beta0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE0feed_forward_sub_net/batch_normalization_3/gamma0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE/feed_forward_sub_net/batch_normalization_3/beta0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE0feed_forward_sub_net/batch_normalization_4/gamma0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE/feed_forward_sub_net/batch_normalization_4/beta0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE!feed_forward_sub_net/dense/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE#feed_forward_sub_net/dense_1/kernel1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE#feed_forward_sub_net/dense_2/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE#feed_forward_sub_net/dense_3/kernel1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE#feed_forward_sub_net/dense_4/kernel1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE!feed_forward_sub_net/dense_4/bias1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE4feed_forward_sub_net/batch_normalization/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE8feed_forward_sub_net/batch_normalization/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE6feed_forward_sub_net/batch_normalization_1/moving_mean'variables/12/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE:feed_forward_sub_net/batch_normalization_1/moving_variance'variables/13/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE6feed_forward_sub_net/batch_normalization_2/moving_mean'variables/14/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE:feed_forward_sub_net/batch_normalization_2/moving_variance'variables/15/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE6feed_forward_sub_net/batch_normalization_3/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE:feed_forward_sub_net/batch_normalization_3/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE6feed_forward_sub_net/batch_normalization_4/moving_mean'variables/18/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE:feed_forward_sub_net/batch_normalization_4/moving_variance'variables/19/.ATTRIBUTES/VARIABLE_VALUE
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
�
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
�
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
�
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
�
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
�
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
�
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
�
Ptrainable_variables

~layers
Qregularization_losses
metrics
R	variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics

0
 

0
�
Ttrainable_variables
�layers
Uregularization_losses
�metrics
V	variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics

 0
 

 0
�
Xtrainable_variables
�layers
Yregularization_losses
�metrics
Z	variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics

!0
"1
 

!0
"1
�
\trainable_variables
�layers
]regularization_losses
�metrics
^	variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
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
:����������*
dtype0*
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_14feed_forward_sub_net/batch_normalization/moving_mean8feed_forward_sub_net/batch_normalization/moving_variance-feed_forward_sub_net/batch_normalization/beta.feed_forward_sub_net/batch_normalization/gamma!feed_forward_sub_net/dense/kernel6feed_forward_sub_net/batch_normalization_1/moving_mean:feed_forward_sub_net/batch_normalization_1/moving_variance/feed_forward_sub_net/batch_normalization_1/beta0feed_forward_sub_net/batch_normalization_1/gamma#feed_forward_sub_net/dense_1/kernel6feed_forward_sub_net/batch_normalization_2/moving_mean:feed_forward_sub_net/batch_normalization_2/moving_variance/feed_forward_sub_net/batch_normalization_2/beta0feed_forward_sub_net/batch_normalization_2/gamma#feed_forward_sub_net/dense_2/kernel6feed_forward_sub_net/batch_normalization_3/moving_mean:feed_forward_sub_net/batch_normalization_3/moving_variance/feed_forward_sub_net/batch_normalization_3/beta0feed_forward_sub_net/batch_normalization_3/gamma#feed_forward_sub_net/dense_3/kernel6feed_forward_sub_net/batch_normalization_4/moving_mean:feed_forward_sub_net/batch_normalization_4/moving_variance/feed_forward_sub_net/batch_normalization_4/beta0feed_forward_sub_net/batch_normalization_4/gamma#feed_forward_sub_net/dense_4/kernel!feed_forward_sub_net/dense_4/bias*&
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_308253
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameBfeed_forward_sub_net/batch_normalization/gamma/Read/ReadVariableOpAfeed_forward_sub_net/batch_normalization/beta/Read/ReadVariableOpDfeed_forward_sub_net/batch_normalization_1/gamma/Read/ReadVariableOpCfeed_forward_sub_net/batch_normalization_1/beta/Read/ReadVariableOpDfeed_forward_sub_net/batch_normalization_2/gamma/Read/ReadVariableOpCfeed_forward_sub_net/batch_normalization_2/beta/Read/ReadVariableOpDfeed_forward_sub_net/batch_normalization_3/gamma/Read/ReadVariableOpCfeed_forward_sub_net/batch_normalization_3/beta/Read/ReadVariableOpDfeed_forward_sub_net/batch_normalization_4/gamma/Read/ReadVariableOpCfeed_forward_sub_net/batch_normalization_4/beta/Read/ReadVariableOp5feed_forward_sub_net/dense/kernel/Read/ReadVariableOp7feed_forward_sub_net/dense_1/kernel/Read/ReadVariableOp7feed_forward_sub_net/dense_2/kernel/Read/ReadVariableOp7feed_forward_sub_net/dense_3/kernel/Read/ReadVariableOp7feed_forward_sub_net/dense_4/kernel/Read/ReadVariableOp5feed_forward_sub_net/dense_4/bias/Read/ReadVariableOpHfeed_forward_sub_net/batch_normalization/moving_mean/Read/ReadVariableOpLfeed_forward_sub_net/batch_normalization/moving_variance/Read/ReadVariableOpJfeed_forward_sub_net/batch_normalization_1/moving_mean/Read/ReadVariableOpNfeed_forward_sub_net/batch_normalization_1/moving_variance/Read/ReadVariableOpJfeed_forward_sub_net/batch_normalization_2/moving_mean/Read/ReadVariableOpNfeed_forward_sub_net/batch_normalization_2/moving_variance/Read/ReadVariableOpJfeed_forward_sub_net/batch_normalization_3/moving_mean/Read/ReadVariableOpNfeed_forward_sub_net/batch_normalization_3/moving_variance/Read/ReadVariableOpJfeed_forward_sub_net/batch_normalization_4/moving_mean/Read/ReadVariableOpNfeed_forward_sub_net/batch_normalization_4/moving_variance/Read/ReadVariableOpConst*'
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
__inference__traced_save_309651
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename.feed_forward_sub_net/batch_normalization/gamma-feed_forward_sub_net/batch_normalization/beta0feed_forward_sub_net/batch_normalization_1/gamma/feed_forward_sub_net/batch_normalization_1/beta0feed_forward_sub_net/batch_normalization_2/gamma/feed_forward_sub_net/batch_normalization_2/beta0feed_forward_sub_net/batch_normalization_3/gamma/feed_forward_sub_net/batch_normalization_3/beta0feed_forward_sub_net/batch_normalization_4/gamma/feed_forward_sub_net/batch_normalization_4/beta!feed_forward_sub_net/dense/kernel#feed_forward_sub_net/dense_1/kernel#feed_forward_sub_net/dense_2/kernel#feed_forward_sub_net/dense_3/kernel#feed_forward_sub_net/dense_4/kernel!feed_forward_sub_net/dense_4/bias4feed_forward_sub_net/batch_normalization/moving_mean8feed_forward_sub_net/batch_normalization/moving_variance6feed_forward_sub_net/batch_normalization_1/moving_mean:feed_forward_sub_net/batch_normalization_1/moving_variance6feed_forward_sub_net/batch_normalization_2/moving_mean:feed_forward_sub_net/batch_normalization_2/moving_variance6feed_forward_sub_net/batch_normalization_3/moving_mean:feed_forward_sub_net/batch_normalization_3/moving_variance6feed_forward_sub_net/batch_normalization_4/moving_mean:feed_forward_sub_net/batch_normalization_4/moving_variance*&
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
"__inference__traced_restore_309739��
�
�
4__inference_batch_normalization_layer_call_fn_309134

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_3067932
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_1_layer_call_fn_309229

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3070212
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
~
(__inference_dense_3_layer_call_fn_309531

inputs
unknown:
��
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_3076852
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:����������: 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�,
�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_307187

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�
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
:	�*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	�2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������2
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
:	�*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
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
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes	
:�2
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
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_1/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:����������2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
A__inference_dense_layer_call_and_return_conditional_losses_309482

inputs2
matmul_readvariableop_resource:
��
identity��MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMull
IdentityIdentityMatMul:product:0^NoOp*
T0*(
_output_shapes
:����������2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:����������: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�C
�
P__inference_feed_forward_sub_net_layer_call_and_return_conditional_losses_307716
x)
batch_normalization_307606:	�)
batch_normalization_307608:	�)
batch_normalization_307610:	�)
batch_normalization_307612:	� 
dense_307623:
��+
batch_normalization_1_307626:	�+
batch_normalization_1_307628:	�+
batch_normalization_1_307630:	�+
batch_normalization_1_307632:	�"
dense_1_307644:
��+
batch_normalization_2_307647:	�+
batch_normalization_2_307649:	�+
batch_normalization_2_307651:	�+
batch_normalization_2_307653:	�"
dense_2_307665:
��+
batch_normalization_3_307668:	�+
batch_normalization_3_307670:	�+
batch_normalization_3_307672:	�+
batch_normalization_3_307674:	�"
dense_3_307686:
��+
batch_normalization_4_307689:	�+
batch_normalization_4_307691:	�+
batch_normalization_4_307693:	�+
batch_normalization_4_307695:	�"
dense_4_307710:
��
dense_4_307712:	�
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�-batch_normalization_4/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCallxbatch_normalization_307606batch_normalization_307608batch_normalization_307610batch_normalization_307612*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_3067932-
+batch_normalization/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_307623*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_3076222
dense/StatefulPartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_1_307626batch_normalization_1_307628batch_normalization_1_307630batch_normalization_1_307632*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3069592/
-batch_normalization_1/StatefulPartitionedCall
ReluRelu6batch_normalization_1/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:����������2
Relu�
dense_1/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0dense_1_307644*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_3076432!
dense_1/StatefulPartitionedCall�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_2_307647batch_normalization_2_307649batch_normalization_2_307651batch_normalization_2_307653*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3071252/
-batch_normalization_2/StatefulPartitionedCall�
Relu_1Relu6batch_normalization_2/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:����������2
Relu_1�
dense_2/StatefulPartitionedCallStatefulPartitionedCallRelu_1:activations:0dense_2_307665*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_3076642!
dense_2/StatefulPartitionedCall�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0batch_normalization_3_307668batch_normalization_3_307670batch_normalization_3_307672batch_normalization_3_307674*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3072912/
-batch_normalization_3/StatefulPartitionedCall�
Relu_2Relu6batch_normalization_3/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:����������2
Relu_2�
dense_3/StatefulPartitionedCallStatefulPartitionedCallRelu_2:activations:0dense_3_307686*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_3076852!
dense_3/StatefulPartitionedCall�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0batch_normalization_4_307689batch_normalization_4_307691batch_normalization_4_307693batch_normalization_4_307695*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_3074572/
-batch_normalization_4/StatefulPartitionedCall�
Relu_3Relu6batch_normalization_4/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:����������2
Relu_3�
dense_4/StatefulPartitionedCallStatefulPartitionedCallRelu_3:activations:0dense_4_307710dense_4_307712*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_3077092!
dense_4/StatefulPartitionedCall�
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identity�
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:����������: : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:K G
(
_output_shapes
:����������

_user_specified_namex
�
�
6__inference_batch_normalization_2_layer_call_fn_309298

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3071252
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_4_layer_call_fn_309462

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_3074572
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
C__inference_dense_1_layer_call_and_return_conditional_losses_307643

inputs2
matmul_readvariableop_resource:
��
identity��MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMull
IdentityIdentityMatMul:product:0^NoOp*
T0*(
_output_shapes
:����������2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:����������: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
4__inference_batch_normalization_layer_call_fn_309147

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_3068552
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_1_layer_call_fn_309216

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3069592
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
5__inference_feed_forward_sub_net_layer_call_fn_309008
x
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�
	unknown_7:	�
	unknown_8:
��
	unknown_9:	�

unknown_10:	�

unknown_11:	�

unknown_12:	�

unknown_13:
��

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:	�

unknown_18:
��

unknown_19:	�

unknown_20:	�

unknown_21:	�

unknown_22:	�

unknown_23:
��

unknown_24:	�
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
 *(
_output_shapes
:����������*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_feed_forward_sub_net_layer_call_and_return_conditional_losses_3079422
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:����������: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:����������

_user_specified_namex
�
�
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_307291

inputs+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�-
cast_2_readvariableop_resource:	�-
cast_3_readvariableop_resource:	�
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOp�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_1/ReadVariableOp�
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_2/ReadVariableOp�
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_3/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:����������2

Identity�
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_307125

inputs+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�-
cast_2_readvariableop_resource:	�-
cast_3_readvariableop_resource:	�
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOp�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_1/ReadVariableOp�
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_2/ReadVariableOp�
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_3/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:����������2

Identity�
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_309249

inputs+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�-
cast_2_readvariableop_resource:	�-
cast_3_readvariableop_resource:	�
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOp�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_1/ReadVariableOp�
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_2/ReadVariableOp�
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_3/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:����������2

Identity�
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�C
�
P__inference_feed_forward_sub_net_layer_call_and_return_conditional_losses_307942
x)
batch_normalization_307875:	�)
batch_normalization_307877:	�)
batch_normalization_307879:	�)
batch_normalization_307881:	� 
dense_307884:
��+
batch_normalization_1_307887:	�+
batch_normalization_1_307889:	�+
batch_normalization_1_307891:	�+
batch_normalization_1_307893:	�"
dense_1_307897:
��+
batch_normalization_2_307900:	�+
batch_normalization_2_307902:	�+
batch_normalization_2_307904:	�+
batch_normalization_2_307906:	�"
dense_2_307910:
��+
batch_normalization_3_307913:	�+
batch_normalization_3_307915:	�+
batch_normalization_3_307917:	�+
batch_normalization_3_307919:	�"
dense_3_307923:
��+
batch_normalization_4_307926:	�+
batch_normalization_4_307928:	�+
batch_normalization_4_307930:	�+
batch_normalization_4_307932:	�"
dense_4_307936:
��
dense_4_307938:	�
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�-batch_normalization_4/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCallxbatch_normalization_307875batch_normalization_307877batch_normalization_307879batch_normalization_307881*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_3068552-
+batch_normalization/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_307884*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_3076222
dense/StatefulPartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_1_307887batch_normalization_1_307889batch_normalization_1_307891batch_normalization_1_307893*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3070212/
-batch_normalization_1/StatefulPartitionedCall
ReluRelu6batch_normalization_1/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:����������2
Relu�
dense_1/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0dense_1_307897*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_3076432!
dense_1/StatefulPartitionedCall�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_2_307900batch_normalization_2_307902batch_normalization_2_307904batch_normalization_2_307906*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3071872/
-batch_normalization_2/StatefulPartitionedCall�
Relu_1Relu6batch_normalization_2/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:����������2
Relu_1�
dense_2/StatefulPartitionedCallStatefulPartitionedCallRelu_1:activations:0dense_2_307910*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_3076642!
dense_2/StatefulPartitionedCall�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0batch_normalization_3_307913batch_normalization_3_307915batch_normalization_3_307917batch_normalization_3_307919*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3073532/
-batch_normalization_3/StatefulPartitionedCall�
Relu_2Relu6batch_normalization_3/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:����������2
Relu_2�
dense_3/StatefulPartitionedCallStatefulPartitionedCallRelu_2:activations:0dense_3_307923*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_3076852!
dense_3/StatefulPartitionedCall�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0batch_normalization_4_307926batch_normalization_4_307928batch_normalization_4_307930batch_normalization_4_307932*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_3075192/
-batch_normalization_4/StatefulPartitionedCall�
Relu_3Relu6batch_normalization_4/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:����������2
Relu_3�
dense_4/StatefulPartitionedCallStatefulPartitionedCallRelu_3:activations:0dense_4_307936dense_4_307938*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_3077092!
dense_4/StatefulPartitionedCall�
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identity�
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:����������: : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:K G
(
_output_shapes
:����������

_user_specified_namex
�,
�
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_309367

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�
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
:	�*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	�2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������2
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
:	�*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
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
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes	
:�2
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
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_1/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:����������2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
C__inference_dense_4_layer_call_and_return_conditional_losses_307709

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddl
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_308253
input_1
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�
	unknown_7:	�
	unknown_8:
��
	unknown_9:	�

unknown_10:	�

unknown_11:	�

unknown_12:	�

unknown_13:
��

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:	�

unknown_18:
��

unknown_19:	�

unknown_20:	�

unknown_21:	�

unknown_22:	�

unknown_23:
��

unknown_24:	�
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
 *(
_output_shapes
:����������*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_3067692
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:����������: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_306793

inputs+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�-
cast_2_readvariableop_resource:	�-
cast_3_readvariableop_resource:	�
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOp�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_1/ReadVariableOp�
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_2/ReadVariableOp�
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_3/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:����������2

Identity�
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�,
�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_309285

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�
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
:	�*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	�2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������2
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
:	�*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
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
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes	
:�2
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
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_1/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:����������2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
C__inference_dense_1_layer_call_and_return_conditional_losses_309496

inputs2
matmul_readvariableop_resource:
��
identity��MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMull
IdentityIdentityMatMul:product:0^NoOp*
T0*(
_output_shapes
:����������2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:����������: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_2_layer_call_fn_309311

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3071872
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
(__inference_dense_4_layer_call_fn_309550

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_3077092
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�,
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_307021

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�
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
:	�*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	�2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������2
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
:	�*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
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
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes	
:�2
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
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_1/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:����������2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_306959

inputs+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�-
cast_2_readvariableop_resource:	�-
cast_3_readvariableop_resource:	�
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOp�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_1/ReadVariableOp�
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_2/ReadVariableOp�
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_3/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:����������2

Identity�
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_309167

inputs+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�-
cast_2_readvariableop_resource:	�-
cast_3_readvariableop_resource:	�
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOp�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_1/ReadVariableOp�
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_2/ReadVariableOp�
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_3/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:����������2

Identity�
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�,
�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_307519

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�
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
:	�*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	�2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������2
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
:	�*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
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
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes	
:�2
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
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_1/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:����������2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_3_layer_call_fn_309393

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3073532
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
C__inference_dense_2_layer_call_and_return_conditional_losses_309510

inputs2
matmul_readvariableop_resource:
��
identity��MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMull
IdentityIdentityMatMul:product:0^NoOp*
T0*(
_output_shapes
:����������2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:����������: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�,
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_306855

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�
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
:	�*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	�2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������2
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
:	�*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
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
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes	
:�2
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
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_1/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:����������2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
5__inference_feed_forward_sub_net_layer_call_fn_308894
input_1
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�
	unknown_7:	�
	unknown_8:
��
	unknown_9:	�

unknown_10:	�

unknown_11:	�

unknown_12:	�

unknown_13:
��

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:	�

unknown_18:
��

unknown_19:	�

unknown_20:	�

unknown_21:	�

unknown_22:	�

unknown_23:
��

unknown_24:	�
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
 *(
_output_shapes
:����������*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_feed_forward_sub_net_layer_call_and_return_conditional_losses_3077162
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:����������: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�
~
(__inference_dense_2_layer_call_fn_309517

inputs
unknown:
��
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_3076642
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:����������: 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
|
&__inference_dense_layer_call_fn_309489

inputs
unknown:
��
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_3076222
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:����������: 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
5__inference_feed_forward_sub_net_layer_call_fn_308951
x
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�
	unknown_7:	�
	unknown_8:
��
	unknown_9:	�

unknown_10:	�

unknown_11:	�

unknown_12:	�

unknown_13:
��

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:	�

unknown_18:
��

unknown_19:	�

unknown_20:	�

unknown_21:	�

unknown_22:	�

unknown_23:
��

unknown_24:	�
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
 *(
_output_shapes
:����������*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_feed_forward_sub_net_layer_call_and_return_conditional_losses_3077162
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:����������: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:����������

_user_specified_namex
�z
�
"__inference__traced_restore_309739
file_prefixN
?assignvariableop_feed_forward_sub_net_batch_normalization_gamma:	�O
@assignvariableop_1_feed_forward_sub_net_batch_normalization_beta:	�R
Cassignvariableop_2_feed_forward_sub_net_batch_normalization_1_gamma:	�Q
Bassignvariableop_3_feed_forward_sub_net_batch_normalization_1_beta:	�R
Cassignvariableop_4_feed_forward_sub_net_batch_normalization_2_gamma:	�Q
Bassignvariableop_5_feed_forward_sub_net_batch_normalization_2_beta:	�R
Cassignvariableop_6_feed_forward_sub_net_batch_normalization_3_gamma:	�Q
Bassignvariableop_7_feed_forward_sub_net_batch_normalization_3_beta:	�R
Cassignvariableop_8_feed_forward_sub_net_batch_normalization_4_gamma:	�Q
Bassignvariableop_9_feed_forward_sub_net_batch_normalization_4_beta:	�I
5assignvariableop_10_feed_forward_sub_net_dense_kernel:
��K
7assignvariableop_11_feed_forward_sub_net_dense_1_kernel:
��K
7assignvariableop_12_feed_forward_sub_net_dense_2_kernel:
��K
7assignvariableop_13_feed_forward_sub_net_dense_3_kernel:
��K
7assignvariableop_14_feed_forward_sub_net_dense_4_kernel:
��D
5assignvariableop_15_feed_forward_sub_net_dense_4_bias:	�W
Hassignvariableop_16_feed_forward_sub_net_batch_normalization_moving_mean:	�[
Lassignvariableop_17_feed_forward_sub_net_batch_normalization_moving_variance:	�Y
Jassignvariableop_18_feed_forward_sub_net_batch_normalization_1_moving_mean:	�]
Nassignvariableop_19_feed_forward_sub_net_batch_normalization_1_moving_variance:	�Y
Jassignvariableop_20_feed_forward_sub_net_batch_normalization_2_moving_mean:	�]
Nassignvariableop_21_feed_forward_sub_net_batch_normalization_2_moving_variance:	�Y
Jassignvariableop_22_feed_forward_sub_net_batch_normalization_3_moving_mean:	�]
Nassignvariableop_23_feed_forward_sub_net_batch_normalization_3_moving_variance:	�Y
Jassignvariableop_24_feed_forward_sub_net_batch_normalization_4_moving_mean:	�]
Nassignvariableop_25_feed_forward_sub_net_batch_normalization_4_moving_variance:	�
identity_27��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�	
value�	B�	B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
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
AssignVariableOpAssignVariableOp?assignvariableop_feed_forward_sub_net_batch_normalization_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp@assignvariableop_1_feed_forward_sub_net_batch_normalization_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpCassignvariableop_2_feed_forward_sub_net_batch_normalization_1_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpBassignvariableop_3_feed_forward_sub_net_batch_normalization_1_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpCassignvariableop_4_feed_forward_sub_net_batch_normalization_2_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpBassignvariableop_5_feed_forward_sub_net_batch_normalization_2_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpCassignvariableop_6_feed_forward_sub_net_batch_normalization_3_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpBassignvariableop_7_feed_forward_sub_net_batch_normalization_3_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpCassignvariableop_8_feed_forward_sub_net_batch_normalization_4_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpBassignvariableop_9_feed_forward_sub_net_batch_normalization_4_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp5assignvariableop_10_feed_forward_sub_net_dense_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp7assignvariableop_11_feed_forward_sub_net_dense_1_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp7assignvariableop_12_feed_forward_sub_net_dense_2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp7assignvariableop_13_feed_forward_sub_net_dense_3_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp7assignvariableop_14_feed_forward_sub_net_dense_4_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp5assignvariableop_15_feed_forward_sub_net_dense_4_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOpHassignvariableop_16_feed_forward_sub_net_batch_normalization_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOpLassignvariableop_17_feed_forward_sub_net_batch_normalization_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOpJassignvariableop_18_feed_forward_sub_net_batch_normalization_1_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOpNassignvariableop_19_feed_forward_sub_net_batch_normalization_1_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOpJassignvariableop_20_feed_forward_sub_net_batch_normalization_2_moving_meanIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOpNassignvariableop_21_feed_forward_sub_net_batch_normalization_2_moving_varianceIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOpJassignvariableop_22_feed_forward_sub_net_batch_normalization_3_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOpNassignvariableop_23_feed_forward_sub_net_batch_normalization_3_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOpJassignvariableop_24_feed_forward_sub_net_batch_normalization_4_moving_meanIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOpNassignvariableop_25_feed_forward_sub_net_batch_normalization_4_moving_varianceIdentity_25:output:0"/device:CPU:0*
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
��
�
P__inference_feed_forward_sub_net_layer_call_and_return_conditional_losses_308359
x?
0batch_normalization_cast_readvariableop_resource:	�A
2batch_normalization_cast_1_readvariableop_resource:	�A
2batch_normalization_cast_2_readvariableop_resource:	�A
2batch_normalization_cast_3_readvariableop_resource:	�8
$dense_matmul_readvariableop_resource:
��A
2batch_normalization_1_cast_readvariableop_resource:	�C
4batch_normalization_1_cast_1_readvariableop_resource:	�C
4batch_normalization_1_cast_2_readvariableop_resource:	�C
4batch_normalization_1_cast_3_readvariableop_resource:	�:
&dense_1_matmul_readvariableop_resource:
��A
2batch_normalization_2_cast_readvariableop_resource:	�C
4batch_normalization_2_cast_1_readvariableop_resource:	�C
4batch_normalization_2_cast_2_readvariableop_resource:	�C
4batch_normalization_2_cast_3_readvariableop_resource:	�:
&dense_2_matmul_readvariableop_resource:
��A
2batch_normalization_3_cast_readvariableop_resource:	�C
4batch_normalization_3_cast_1_readvariableop_resource:	�C
4batch_normalization_3_cast_2_readvariableop_resource:	�C
4batch_normalization_3_cast_3_readvariableop_resource:	�:
&dense_3_matmul_readvariableop_resource:
��A
2batch_normalization_4_cast_readvariableop_resource:	�C
4batch_normalization_4_cast_1_readvariableop_resource:	�C
4batch_normalization_4_cast_2_readvariableop_resource:	�C
4batch_normalization_4_cast_3_readvariableop_resource:	�:
&dense_4_matmul_readvariableop_resource:
��6
'dense_4_biasadd_readvariableop_resource:	�
identity��'batch_normalization/Cast/ReadVariableOp�)batch_normalization/Cast_1/ReadVariableOp�)batch_normalization/Cast_2/ReadVariableOp�)batch_normalization/Cast_3/ReadVariableOp�)batch_normalization_1/Cast/ReadVariableOp�+batch_normalization_1/Cast_1/ReadVariableOp�+batch_normalization_1/Cast_2/ReadVariableOp�+batch_normalization_1/Cast_3/ReadVariableOp�)batch_normalization_2/Cast/ReadVariableOp�+batch_normalization_2/Cast_1/ReadVariableOp�+batch_normalization_2/Cast_2/ReadVariableOp�+batch_normalization_2/Cast_3/ReadVariableOp�)batch_normalization_3/Cast/ReadVariableOp�+batch_normalization_3/Cast_1/ReadVariableOp�+batch_normalization_3/Cast_2/ReadVariableOp�+batch_normalization_3/Cast_3/ReadVariableOp�)batch_normalization_4/Cast/ReadVariableOp�+batch_normalization_4/Cast_1/ReadVariableOp�+batch_normalization_4/Cast_2/ReadVariableOp�+batch_normalization_4/Cast_3/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/MatMul/ReadVariableOp�dense_3/MatMul/ReadVariableOp�dense_4/BiasAdd/ReadVariableOp�dense_4/MatMul/ReadVariableOp�
'batch_normalization/Cast/ReadVariableOpReadVariableOp0batch_normalization_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02)
'batch_normalization/Cast/ReadVariableOp�
)batch_normalization/Cast_1/ReadVariableOpReadVariableOp2batch_normalization_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization/Cast_1/ReadVariableOp�
)batch_normalization/Cast_2/ReadVariableOpReadVariableOp2batch_normalization_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization/Cast_2/ReadVariableOp�
)batch_normalization/Cast_3/ReadVariableOpReadVariableOp2batch_normalization_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization/Cast_3/ReadVariableOp�
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2%
#batch_normalization/batchnorm/add/y�
!batch_normalization/batchnorm/addAddV21batch_normalization/Cast_1/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2#
!batch_normalization/batchnorm/add�
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization/batchnorm/Rsqrt�
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:01batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2#
!batch_normalization/batchnorm/mul�
#batch_normalization/batchnorm/mul_1Mulx%batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2%
#batch_normalization/batchnorm/mul_1�
#batch_normalization/batchnorm/mul_2Mul/batch_normalization/Cast/ReadVariableOp:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization/batchnorm/mul_2�
!batch_normalization/batchnorm/subSub1batch_normalization/Cast_2/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2#
!batch_normalization/batchnorm/sub�
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2%
#batch_normalization/batchnorm/add_1�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMul'batch_normalization/batchnorm/add_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/MatMul�
)batch_normalization_1/Cast/ReadVariableOpReadVariableOp2batch_normalization_1_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_1/Cast/ReadVariableOp�
+batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_1/Cast_1/ReadVariableOp�
+batch_normalization_1/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_1_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_1/Cast_2/ReadVariableOp�
+batch_normalization_1/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_1_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_1/Cast_3/ReadVariableOp�
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2'
%batch_normalization_1/batchnorm/add/y�
#batch_normalization_1/batchnorm/addAddV23batch_normalization_1/Cast_1/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_1/batchnorm/add�
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_1/batchnorm/Rsqrt�
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:03batch_normalization_1/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_1/batchnorm/mul�
%batch_normalization_1/batchnorm/mul_1Muldense/MatMul:product:0'batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_1/batchnorm/mul_1�
%batch_normalization_1/batchnorm/mul_2Mul1batch_normalization_1/Cast/ReadVariableOp:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_1/batchnorm/mul_2�
#batch_normalization_1/batchnorm/subSub3batch_normalization_1/Cast_2/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_1/batchnorm/sub�
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_1/batchnorm/add_1r
ReluRelu)batch_normalization_1/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Relu�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMulRelu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_1/MatMul�
)batch_normalization_2/Cast/ReadVariableOpReadVariableOp2batch_normalization_2_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_2/Cast/ReadVariableOp�
+batch_normalization_2/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_2/Cast_1/ReadVariableOp�
+batch_normalization_2/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_2_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_2/Cast_2/ReadVariableOp�
+batch_normalization_2/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_2_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_2/Cast_3/ReadVariableOp�
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2'
%batch_normalization_2/batchnorm/add/y�
#batch_normalization_2/batchnorm/addAddV23batch_normalization_2/Cast_1/ReadVariableOp:value:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_2/batchnorm/add�
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_2/batchnorm/Rsqrt�
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:03batch_normalization_2/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_2/batchnorm/mul�
%batch_normalization_2/batchnorm/mul_1Muldense_1/MatMul:product:0'batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_2/batchnorm/mul_1�
%batch_normalization_2/batchnorm/mul_2Mul1batch_normalization_2/Cast/ReadVariableOp:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_2/batchnorm/mul_2�
#batch_normalization_2/batchnorm/subSub3batch_normalization_2/Cast_2/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_2/batchnorm/sub�
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_2/batchnorm/add_1v
Relu_1Relu)batch_normalization_2/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Relu_1�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_2/MatMul/ReadVariableOp�
dense_2/MatMulMatMulRelu_1:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_2/MatMul�
)batch_normalization_3/Cast/ReadVariableOpReadVariableOp2batch_normalization_3_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_3/Cast/ReadVariableOp�
+batch_normalization_3/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_3_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_3/Cast_1/ReadVariableOp�
+batch_normalization_3/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_3_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_3/Cast_2/ReadVariableOp�
+batch_normalization_3/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_3_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_3/Cast_3/ReadVariableOp�
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2'
%batch_normalization_3/batchnorm/add/y�
#batch_normalization_3/batchnorm/addAddV23batch_normalization_3/Cast_1/ReadVariableOp:value:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_3/batchnorm/add�
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_3/batchnorm/Rsqrt�
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:03batch_normalization_3/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_3/batchnorm/mul�
%batch_normalization_3/batchnorm/mul_1Muldense_2/MatMul:product:0'batch_normalization_3/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_3/batchnorm/mul_1�
%batch_normalization_3/batchnorm/mul_2Mul1batch_normalization_3/Cast/ReadVariableOp:value:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_3/batchnorm/mul_2�
#batch_normalization_3/batchnorm/subSub3batch_normalization_3/Cast_2/ReadVariableOp:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_3/batchnorm/sub�
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_3/batchnorm/add_1v
Relu_2Relu)batch_normalization_3/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Relu_2�
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_3/MatMul/ReadVariableOp�
dense_3/MatMulMatMulRelu_2:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_3/MatMul�
)batch_normalization_4/Cast/ReadVariableOpReadVariableOp2batch_normalization_4_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_4/Cast/ReadVariableOp�
+batch_normalization_4/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_4_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_4/Cast_1/ReadVariableOp�
+batch_normalization_4/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_4_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_4/Cast_2/ReadVariableOp�
+batch_normalization_4/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_4_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_4/Cast_3/ReadVariableOp�
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2'
%batch_normalization_4/batchnorm/add/y�
#batch_normalization_4/batchnorm/addAddV23batch_normalization_4/Cast_1/ReadVariableOp:value:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_4/batchnorm/add�
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_4/batchnorm/Rsqrt�
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:03batch_normalization_4/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_4/batchnorm/mul�
%batch_normalization_4/batchnorm/mul_1Muldense_3/MatMul:product:0'batch_normalization_4/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_4/batchnorm/mul_1�
%batch_normalization_4/batchnorm/mul_2Mul1batch_normalization_4/Cast/ReadVariableOp:value:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_4/batchnorm/mul_2�
#batch_normalization_4/batchnorm/subSub3batch_normalization_4/Cast_2/ReadVariableOp:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_4/batchnorm/sub�
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_4/batchnorm/add_1v
Relu_3Relu)batch_normalization_4/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Relu_3�
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_4/MatMul/ReadVariableOp�
dense_4/MatMulMatMulRelu_3:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_4/MatMul�
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_4/BiasAdd/ReadVariableOp�
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_4/BiasAddt
IdentityIdentitydense_4/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identity�	
NoOpNoOp(^batch_normalization/Cast/ReadVariableOp*^batch_normalization/Cast_1/ReadVariableOp*^batch_normalization/Cast_2/ReadVariableOp*^batch_normalization/Cast_3/ReadVariableOp*^batch_normalization_1/Cast/ReadVariableOp,^batch_normalization_1/Cast_1/ReadVariableOp,^batch_normalization_1/Cast_2/ReadVariableOp,^batch_normalization_1/Cast_3/ReadVariableOp*^batch_normalization_2/Cast/ReadVariableOp,^batch_normalization_2/Cast_1/ReadVariableOp,^batch_normalization_2/Cast_2/ReadVariableOp,^batch_normalization_2/Cast_3/ReadVariableOp*^batch_normalization_3/Cast/ReadVariableOp,^batch_normalization_3/Cast_1/ReadVariableOp,^batch_normalization_3/Cast_2/ReadVariableOp,^batch_normalization_3/Cast_3/ReadVariableOp*^batch_normalization_4/Cast/ReadVariableOp,^batch_normalization_4/Cast_1/ReadVariableOp,^batch_normalization_4/Cast_2/ReadVariableOp,^batch_normalization_4/Cast_3/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:����������: : : : : : : : : : : : : : : : : : : : : : : : : : 2R
'batch_normalization/Cast/ReadVariableOp'batch_normalization/Cast/ReadVariableOp2V
)batch_normalization/Cast_1/ReadVariableOp)batch_normalization/Cast_1/ReadVariableOp2V
)batch_normalization/Cast_2/ReadVariableOp)batch_normalization/Cast_2/ReadVariableOp2V
)batch_normalization/Cast_3/ReadVariableOp)batch_normalization/Cast_3/ReadVariableOp2V
)batch_normalization_1/Cast/ReadVariableOp)batch_normalization_1/Cast/ReadVariableOp2Z
+batch_normalization_1/Cast_1/ReadVariableOp+batch_normalization_1/Cast_1/ReadVariableOp2Z
+batch_normalization_1/Cast_2/ReadVariableOp+batch_normalization_1/Cast_2/ReadVariableOp2Z
+batch_normalization_1/Cast_3/ReadVariableOp+batch_normalization_1/Cast_3/ReadVariableOp2V
)batch_normalization_2/Cast/ReadVariableOp)batch_normalization_2/Cast/ReadVariableOp2Z
+batch_normalization_2/Cast_1/ReadVariableOp+batch_normalization_2/Cast_1/ReadVariableOp2Z
+batch_normalization_2/Cast_2/ReadVariableOp+batch_normalization_2/Cast_2/ReadVariableOp2Z
+batch_normalization_2/Cast_3/ReadVariableOp+batch_normalization_2/Cast_3/ReadVariableOp2V
)batch_normalization_3/Cast/ReadVariableOp)batch_normalization_3/Cast/ReadVariableOp2Z
+batch_normalization_3/Cast_1/ReadVariableOp+batch_normalization_3/Cast_1/ReadVariableOp2Z
+batch_normalization_3/Cast_2/ReadVariableOp+batch_normalization_3/Cast_2/ReadVariableOp2Z
+batch_normalization_3/Cast_3/ReadVariableOp+batch_normalization_3/Cast_3/ReadVariableOp2V
)batch_normalization_4/Cast/ReadVariableOp)batch_normalization_4/Cast/ReadVariableOp2Z
+batch_normalization_4/Cast_1/ReadVariableOp+batch_normalization_4/Cast_1/ReadVariableOp2Z
+batch_normalization_4/Cast_2/ReadVariableOp+batch_normalization_4/Cast_2/ReadVariableOp2Z
+batch_normalization_4/Cast_3/ReadVariableOp+batch_normalization_4/Cast_3/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp:K G
(
_output_shapes
:����������

_user_specified_namex
�
�
C__inference_dense_2_layer_call_and_return_conditional_losses_307664

inputs2
matmul_readvariableop_resource:
��
identity��MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMull
IdentityIdentityMatMul:product:0^NoOp*
T0*(
_output_shapes
:����������2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:����������: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�,
�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_309449

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�
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
:	�*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	�2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������2
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
:	�*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
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
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes	
:�2
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
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_1/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:����������2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_309413

inputs+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�-
cast_2_readvariableop_resource:	�-
cast_3_readvariableop_resource:	�
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOp�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_1/ReadVariableOp�
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_2/ReadVariableOp�
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_3/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:����������2

Identity�
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
~
(__inference_dense_1_layer_call_fn_309503

inputs
unknown:
��
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_3076432
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:����������: 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
!__inference__wrapped_model_306769
input_1T
Efeed_forward_sub_net_batch_normalization_cast_readvariableop_resource:	�V
Gfeed_forward_sub_net_batch_normalization_cast_1_readvariableop_resource:	�V
Gfeed_forward_sub_net_batch_normalization_cast_2_readvariableop_resource:	�V
Gfeed_forward_sub_net_batch_normalization_cast_3_readvariableop_resource:	�M
9feed_forward_sub_net_dense_matmul_readvariableop_resource:
��V
Gfeed_forward_sub_net_batch_normalization_1_cast_readvariableop_resource:	�X
Ifeed_forward_sub_net_batch_normalization_1_cast_1_readvariableop_resource:	�X
Ifeed_forward_sub_net_batch_normalization_1_cast_2_readvariableop_resource:	�X
Ifeed_forward_sub_net_batch_normalization_1_cast_3_readvariableop_resource:	�O
;feed_forward_sub_net_dense_1_matmul_readvariableop_resource:
��V
Gfeed_forward_sub_net_batch_normalization_2_cast_readvariableop_resource:	�X
Ifeed_forward_sub_net_batch_normalization_2_cast_1_readvariableop_resource:	�X
Ifeed_forward_sub_net_batch_normalization_2_cast_2_readvariableop_resource:	�X
Ifeed_forward_sub_net_batch_normalization_2_cast_3_readvariableop_resource:	�O
;feed_forward_sub_net_dense_2_matmul_readvariableop_resource:
��V
Gfeed_forward_sub_net_batch_normalization_3_cast_readvariableop_resource:	�X
Ifeed_forward_sub_net_batch_normalization_3_cast_1_readvariableop_resource:	�X
Ifeed_forward_sub_net_batch_normalization_3_cast_2_readvariableop_resource:	�X
Ifeed_forward_sub_net_batch_normalization_3_cast_3_readvariableop_resource:	�O
;feed_forward_sub_net_dense_3_matmul_readvariableop_resource:
��V
Gfeed_forward_sub_net_batch_normalization_4_cast_readvariableop_resource:	�X
Ifeed_forward_sub_net_batch_normalization_4_cast_1_readvariableop_resource:	�X
Ifeed_forward_sub_net_batch_normalization_4_cast_2_readvariableop_resource:	�X
Ifeed_forward_sub_net_batch_normalization_4_cast_3_readvariableop_resource:	�O
;feed_forward_sub_net_dense_4_matmul_readvariableop_resource:
��K
<feed_forward_sub_net_dense_4_biasadd_readvariableop_resource:	�
identity��<feed_forward_sub_net/batch_normalization/Cast/ReadVariableOp�>feed_forward_sub_net/batch_normalization/Cast_1/ReadVariableOp�>feed_forward_sub_net/batch_normalization/Cast_2/ReadVariableOp�>feed_forward_sub_net/batch_normalization/Cast_3/ReadVariableOp�>feed_forward_sub_net/batch_normalization_1/Cast/ReadVariableOp�@feed_forward_sub_net/batch_normalization_1/Cast_1/ReadVariableOp�@feed_forward_sub_net/batch_normalization_1/Cast_2/ReadVariableOp�@feed_forward_sub_net/batch_normalization_1/Cast_3/ReadVariableOp�>feed_forward_sub_net/batch_normalization_2/Cast/ReadVariableOp�@feed_forward_sub_net/batch_normalization_2/Cast_1/ReadVariableOp�@feed_forward_sub_net/batch_normalization_2/Cast_2/ReadVariableOp�@feed_forward_sub_net/batch_normalization_2/Cast_3/ReadVariableOp�>feed_forward_sub_net/batch_normalization_3/Cast/ReadVariableOp�@feed_forward_sub_net/batch_normalization_3/Cast_1/ReadVariableOp�@feed_forward_sub_net/batch_normalization_3/Cast_2/ReadVariableOp�@feed_forward_sub_net/batch_normalization_3/Cast_3/ReadVariableOp�>feed_forward_sub_net/batch_normalization_4/Cast/ReadVariableOp�@feed_forward_sub_net/batch_normalization_4/Cast_1/ReadVariableOp�@feed_forward_sub_net/batch_normalization_4/Cast_2/ReadVariableOp�@feed_forward_sub_net/batch_normalization_4/Cast_3/ReadVariableOp�0feed_forward_sub_net/dense/MatMul/ReadVariableOp�2feed_forward_sub_net/dense_1/MatMul/ReadVariableOp�2feed_forward_sub_net/dense_2/MatMul/ReadVariableOp�2feed_forward_sub_net/dense_3/MatMul/ReadVariableOp�3feed_forward_sub_net/dense_4/BiasAdd/ReadVariableOp�2feed_forward_sub_net/dense_4/MatMul/ReadVariableOp�
<feed_forward_sub_net/batch_normalization/Cast/ReadVariableOpReadVariableOpEfeed_forward_sub_net_batch_normalization_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02>
<feed_forward_sub_net/batch_normalization/Cast/ReadVariableOp�
>feed_forward_sub_net/batch_normalization/Cast_1/ReadVariableOpReadVariableOpGfeed_forward_sub_net_batch_normalization_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02@
>feed_forward_sub_net/batch_normalization/Cast_1/ReadVariableOp�
>feed_forward_sub_net/batch_normalization/Cast_2/ReadVariableOpReadVariableOpGfeed_forward_sub_net_batch_normalization_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02@
>feed_forward_sub_net/batch_normalization/Cast_2/ReadVariableOp�
>feed_forward_sub_net/batch_normalization/Cast_3/ReadVariableOpReadVariableOpGfeed_forward_sub_net_batch_normalization_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02@
>feed_forward_sub_net/batch_normalization/Cast_3/ReadVariableOp�
8feed_forward_sub_net/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2:
8feed_forward_sub_net/batch_normalization/batchnorm/add/y�
6feed_forward_sub_net/batch_normalization/batchnorm/addAddV2Ffeed_forward_sub_net/batch_normalization/Cast_1/ReadVariableOp:value:0Afeed_forward_sub_net/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�28
6feed_forward_sub_net/batch_normalization/batchnorm/add�
8feed_forward_sub_net/batch_normalization/batchnorm/RsqrtRsqrt:feed_forward_sub_net/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:�2:
8feed_forward_sub_net/batch_normalization/batchnorm/Rsqrt�
6feed_forward_sub_net/batch_normalization/batchnorm/mulMul<feed_forward_sub_net/batch_normalization/batchnorm/Rsqrt:y:0Ffeed_forward_sub_net/batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�28
6feed_forward_sub_net/batch_normalization/batchnorm/mul�
8feed_forward_sub_net/batch_normalization/batchnorm/mul_1Mulinput_1:feed_forward_sub_net/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2:
8feed_forward_sub_net/batch_normalization/batchnorm/mul_1�
8feed_forward_sub_net/batch_normalization/batchnorm/mul_2MulDfeed_forward_sub_net/batch_normalization/Cast/ReadVariableOp:value:0:feed_forward_sub_net/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2:
8feed_forward_sub_net/batch_normalization/batchnorm/mul_2�
6feed_forward_sub_net/batch_normalization/batchnorm/subSubFfeed_forward_sub_net/batch_normalization/Cast_2/ReadVariableOp:value:0<feed_forward_sub_net/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�28
6feed_forward_sub_net/batch_normalization/batchnorm/sub�
8feed_forward_sub_net/batch_normalization/batchnorm/add_1AddV2<feed_forward_sub_net/batch_normalization/batchnorm/mul_1:z:0:feed_forward_sub_net/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2:
8feed_forward_sub_net/batch_normalization/batchnorm/add_1�
0feed_forward_sub_net/dense/MatMul/ReadVariableOpReadVariableOp9feed_forward_sub_net_dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype022
0feed_forward_sub_net/dense/MatMul/ReadVariableOp�
!feed_forward_sub_net/dense/MatMulMatMul<feed_forward_sub_net/batch_normalization/batchnorm/add_1:z:08feed_forward_sub_net/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2#
!feed_forward_sub_net/dense/MatMul�
>feed_forward_sub_net/batch_normalization_1/Cast/ReadVariableOpReadVariableOpGfeed_forward_sub_net_batch_normalization_1_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02@
>feed_forward_sub_net/batch_normalization_1/Cast/ReadVariableOp�
@feed_forward_sub_net/batch_normalization_1/Cast_1/ReadVariableOpReadVariableOpIfeed_forward_sub_net_batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02B
@feed_forward_sub_net/batch_normalization_1/Cast_1/ReadVariableOp�
@feed_forward_sub_net/batch_normalization_1/Cast_2/ReadVariableOpReadVariableOpIfeed_forward_sub_net_batch_normalization_1_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02B
@feed_forward_sub_net/batch_normalization_1/Cast_2/ReadVariableOp�
@feed_forward_sub_net/batch_normalization_1/Cast_3/ReadVariableOpReadVariableOpIfeed_forward_sub_net_batch_normalization_1_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02B
@feed_forward_sub_net/batch_normalization_1/Cast_3/ReadVariableOp�
:feed_forward_sub_net/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2<
:feed_forward_sub_net/batch_normalization_1/batchnorm/add/y�
8feed_forward_sub_net/batch_normalization_1/batchnorm/addAddV2Hfeed_forward_sub_net/batch_normalization_1/Cast_1/ReadVariableOp:value:0Cfeed_forward_sub_net/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2:
8feed_forward_sub_net/batch_normalization_1/batchnorm/add�
:feed_forward_sub_net/batch_normalization_1/batchnorm/RsqrtRsqrt<feed_forward_sub_net/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:�2<
:feed_forward_sub_net/batch_normalization_1/batchnorm/Rsqrt�
8feed_forward_sub_net/batch_normalization_1/batchnorm/mulMul>feed_forward_sub_net/batch_normalization_1/batchnorm/Rsqrt:y:0Hfeed_forward_sub_net/batch_normalization_1/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2:
8feed_forward_sub_net/batch_normalization_1/batchnorm/mul�
:feed_forward_sub_net/batch_normalization_1/batchnorm/mul_1Mul+feed_forward_sub_net/dense/MatMul:product:0<feed_forward_sub_net/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2<
:feed_forward_sub_net/batch_normalization_1/batchnorm/mul_1�
:feed_forward_sub_net/batch_normalization_1/batchnorm/mul_2MulFfeed_forward_sub_net/batch_normalization_1/Cast/ReadVariableOp:value:0<feed_forward_sub_net/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2<
:feed_forward_sub_net/batch_normalization_1/batchnorm/mul_2�
8feed_forward_sub_net/batch_normalization_1/batchnorm/subSubHfeed_forward_sub_net/batch_normalization_1/Cast_2/ReadVariableOp:value:0>feed_forward_sub_net/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2:
8feed_forward_sub_net/batch_normalization_1/batchnorm/sub�
:feed_forward_sub_net/batch_normalization_1/batchnorm/add_1AddV2>feed_forward_sub_net/batch_normalization_1/batchnorm/mul_1:z:0<feed_forward_sub_net/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2<
:feed_forward_sub_net/batch_normalization_1/batchnorm/add_1�
feed_forward_sub_net/ReluRelu>feed_forward_sub_net/batch_normalization_1/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
feed_forward_sub_net/Relu�
2feed_forward_sub_net/dense_1/MatMul/ReadVariableOpReadVariableOp;feed_forward_sub_net_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype024
2feed_forward_sub_net/dense_1/MatMul/ReadVariableOp�
#feed_forward_sub_net/dense_1/MatMulMatMul'feed_forward_sub_net/Relu:activations:0:feed_forward_sub_net/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#feed_forward_sub_net/dense_1/MatMul�
>feed_forward_sub_net/batch_normalization_2/Cast/ReadVariableOpReadVariableOpGfeed_forward_sub_net_batch_normalization_2_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02@
>feed_forward_sub_net/batch_normalization_2/Cast/ReadVariableOp�
@feed_forward_sub_net/batch_normalization_2/Cast_1/ReadVariableOpReadVariableOpIfeed_forward_sub_net_batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02B
@feed_forward_sub_net/batch_normalization_2/Cast_1/ReadVariableOp�
@feed_forward_sub_net/batch_normalization_2/Cast_2/ReadVariableOpReadVariableOpIfeed_forward_sub_net_batch_normalization_2_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02B
@feed_forward_sub_net/batch_normalization_2/Cast_2/ReadVariableOp�
@feed_forward_sub_net/batch_normalization_2/Cast_3/ReadVariableOpReadVariableOpIfeed_forward_sub_net_batch_normalization_2_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02B
@feed_forward_sub_net/batch_normalization_2/Cast_3/ReadVariableOp�
:feed_forward_sub_net/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2<
:feed_forward_sub_net/batch_normalization_2/batchnorm/add/y�
8feed_forward_sub_net/batch_normalization_2/batchnorm/addAddV2Hfeed_forward_sub_net/batch_normalization_2/Cast_1/ReadVariableOp:value:0Cfeed_forward_sub_net/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2:
8feed_forward_sub_net/batch_normalization_2/batchnorm/add�
:feed_forward_sub_net/batch_normalization_2/batchnorm/RsqrtRsqrt<feed_forward_sub_net/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:�2<
:feed_forward_sub_net/batch_normalization_2/batchnorm/Rsqrt�
8feed_forward_sub_net/batch_normalization_2/batchnorm/mulMul>feed_forward_sub_net/batch_normalization_2/batchnorm/Rsqrt:y:0Hfeed_forward_sub_net/batch_normalization_2/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2:
8feed_forward_sub_net/batch_normalization_2/batchnorm/mul�
:feed_forward_sub_net/batch_normalization_2/batchnorm/mul_1Mul-feed_forward_sub_net/dense_1/MatMul:product:0<feed_forward_sub_net/batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2<
:feed_forward_sub_net/batch_normalization_2/batchnorm/mul_1�
:feed_forward_sub_net/batch_normalization_2/batchnorm/mul_2MulFfeed_forward_sub_net/batch_normalization_2/Cast/ReadVariableOp:value:0<feed_forward_sub_net/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2<
:feed_forward_sub_net/batch_normalization_2/batchnorm/mul_2�
8feed_forward_sub_net/batch_normalization_2/batchnorm/subSubHfeed_forward_sub_net/batch_normalization_2/Cast_2/ReadVariableOp:value:0>feed_forward_sub_net/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2:
8feed_forward_sub_net/batch_normalization_2/batchnorm/sub�
:feed_forward_sub_net/batch_normalization_2/batchnorm/add_1AddV2>feed_forward_sub_net/batch_normalization_2/batchnorm/mul_1:z:0<feed_forward_sub_net/batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2<
:feed_forward_sub_net/batch_normalization_2/batchnorm/add_1�
feed_forward_sub_net/Relu_1Relu>feed_forward_sub_net/batch_normalization_2/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
feed_forward_sub_net/Relu_1�
2feed_forward_sub_net/dense_2/MatMul/ReadVariableOpReadVariableOp;feed_forward_sub_net_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype024
2feed_forward_sub_net/dense_2/MatMul/ReadVariableOp�
#feed_forward_sub_net/dense_2/MatMulMatMul)feed_forward_sub_net/Relu_1:activations:0:feed_forward_sub_net/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#feed_forward_sub_net/dense_2/MatMul�
>feed_forward_sub_net/batch_normalization_3/Cast/ReadVariableOpReadVariableOpGfeed_forward_sub_net_batch_normalization_3_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02@
>feed_forward_sub_net/batch_normalization_3/Cast/ReadVariableOp�
@feed_forward_sub_net/batch_normalization_3/Cast_1/ReadVariableOpReadVariableOpIfeed_forward_sub_net_batch_normalization_3_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02B
@feed_forward_sub_net/batch_normalization_3/Cast_1/ReadVariableOp�
@feed_forward_sub_net/batch_normalization_3/Cast_2/ReadVariableOpReadVariableOpIfeed_forward_sub_net_batch_normalization_3_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02B
@feed_forward_sub_net/batch_normalization_3/Cast_2/ReadVariableOp�
@feed_forward_sub_net/batch_normalization_3/Cast_3/ReadVariableOpReadVariableOpIfeed_forward_sub_net_batch_normalization_3_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02B
@feed_forward_sub_net/batch_normalization_3/Cast_3/ReadVariableOp�
:feed_forward_sub_net/batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2<
:feed_forward_sub_net/batch_normalization_3/batchnorm/add/y�
8feed_forward_sub_net/batch_normalization_3/batchnorm/addAddV2Hfeed_forward_sub_net/batch_normalization_3/Cast_1/ReadVariableOp:value:0Cfeed_forward_sub_net/batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2:
8feed_forward_sub_net/batch_normalization_3/batchnorm/add�
:feed_forward_sub_net/batch_normalization_3/batchnorm/RsqrtRsqrt<feed_forward_sub_net/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes	
:�2<
:feed_forward_sub_net/batch_normalization_3/batchnorm/Rsqrt�
8feed_forward_sub_net/batch_normalization_3/batchnorm/mulMul>feed_forward_sub_net/batch_normalization_3/batchnorm/Rsqrt:y:0Hfeed_forward_sub_net/batch_normalization_3/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2:
8feed_forward_sub_net/batch_normalization_3/batchnorm/mul�
:feed_forward_sub_net/batch_normalization_3/batchnorm/mul_1Mul-feed_forward_sub_net/dense_2/MatMul:product:0<feed_forward_sub_net/batch_normalization_3/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2<
:feed_forward_sub_net/batch_normalization_3/batchnorm/mul_1�
:feed_forward_sub_net/batch_normalization_3/batchnorm/mul_2MulFfeed_forward_sub_net/batch_normalization_3/Cast/ReadVariableOp:value:0<feed_forward_sub_net/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2<
:feed_forward_sub_net/batch_normalization_3/batchnorm/mul_2�
8feed_forward_sub_net/batch_normalization_3/batchnorm/subSubHfeed_forward_sub_net/batch_normalization_3/Cast_2/ReadVariableOp:value:0>feed_forward_sub_net/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2:
8feed_forward_sub_net/batch_normalization_3/batchnorm/sub�
:feed_forward_sub_net/batch_normalization_3/batchnorm/add_1AddV2>feed_forward_sub_net/batch_normalization_3/batchnorm/mul_1:z:0<feed_forward_sub_net/batch_normalization_3/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2<
:feed_forward_sub_net/batch_normalization_3/batchnorm/add_1�
feed_forward_sub_net/Relu_2Relu>feed_forward_sub_net/batch_normalization_3/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
feed_forward_sub_net/Relu_2�
2feed_forward_sub_net/dense_3/MatMul/ReadVariableOpReadVariableOp;feed_forward_sub_net_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype024
2feed_forward_sub_net/dense_3/MatMul/ReadVariableOp�
#feed_forward_sub_net/dense_3/MatMulMatMul)feed_forward_sub_net/Relu_2:activations:0:feed_forward_sub_net/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#feed_forward_sub_net/dense_3/MatMul�
>feed_forward_sub_net/batch_normalization_4/Cast/ReadVariableOpReadVariableOpGfeed_forward_sub_net_batch_normalization_4_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02@
>feed_forward_sub_net/batch_normalization_4/Cast/ReadVariableOp�
@feed_forward_sub_net/batch_normalization_4/Cast_1/ReadVariableOpReadVariableOpIfeed_forward_sub_net_batch_normalization_4_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02B
@feed_forward_sub_net/batch_normalization_4/Cast_1/ReadVariableOp�
@feed_forward_sub_net/batch_normalization_4/Cast_2/ReadVariableOpReadVariableOpIfeed_forward_sub_net_batch_normalization_4_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02B
@feed_forward_sub_net/batch_normalization_4/Cast_2/ReadVariableOp�
@feed_forward_sub_net/batch_normalization_4/Cast_3/ReadVariableOpReadVariableOpIfeed_forward_sub_net_batch_normalization_4_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02B
@feed_forward_sub_net/batch_normalization_4/Cast_3/ReadVariableOp�
:feed_forward_sub_net/batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2<
:feed_forward_sub_net/batch_normalization_4/batchnorm/add/y�
8feed_forward_sub_net/batch_normalization_4/batchnorm/addAddV2Hfeed_forward_sub_net/batch_normalization_4/Cast_1/ReadVariableOp:value:0Cfeed_forward_sub_net/batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2:
8feed_forward_sub_net/batch_normalization_4/batchnorm/add�
:feed_forward_sub_net/batch_normalization_4/batchnorm/RsqrtRsqrt<feed_forward_sub_net/batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:�2<
:feed_forward_sub_net/batch_normalization_4/batchnorm/Rsqrt�
8feed_forward_sub_net/batch_normalization_4/batchnorm/mulMul>feed_forward_sub_net/batch_normalization_4/batchnorm/Rsqrt:y:0Hfeed_forward_sub_net/batch_normalization_4/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2:
8feed_forward_sub_net/batch_normalization_4/batchnorm/mul�
:feed_forward_sub_net/batch_normalization_4/batchnorm/mul_1Mul-feed_forward_sub_net/dense_3/MatMul:product:0<feed_forward_sub_net/batch_normalization_4/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2<
:feed_forward_sub_net/batch_normalization_4/batchnorm/mul_1�
:feed_forward_sub_net/batch_normalization_4/batchnorm/mul_2MulFfeed_forward_sub_net/batch_normalization_4/Cast/ReadVariableOp:value:0<feed_forward_sub_net/batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2<
:feed_forward_sub_net/batch_normalization_4/batchnorm/mul_2�
8feed_forward_sub_net/batch_normalization_4/batchnorm/subSubHfeed_forward_sub_net/batch_normalization_4/Cast_2/ReadVariableOp:value:0>feed_forward_sub_net/batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2:
8feed_forward_sub_net/batch_normalization_4/batchnorm/sub�
:feed_forward_sub_net/batch_normalization_4/batchnorm/add_1AddV2>feed_forward_sub_net/batch_normalization_4/batchnorm/mul_1:z:0<feed_forward_sub_net/batch_normalization_4/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2<
:feed_forward_sub_net/batch_normalization_4/batchnorm/add_1�
feed_forward_sub_net/Relu_3Relu>feed_forward_sub_net/batch_normalization_4/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
feed_forward_sub_net/Relu_3�
2feed_forward_sub_net/dense_4/MatMul/ReadVariableOpReadVariableOp;feed_forward_sub_net_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype024
2feed_forward_sub_net/dense_4/MatMul/ReadVariableOp�
#feed_forward_sub_net/dense_4/MatMulMatMul)feed_forward_sub_net/Relu_3:activations:0:feed_forward_sub_net/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#feed_forward_sub_net/dense_4/MatMul�
3feed_forward_sub_net/dense_4/BiasAdd/ReadVariableOpReadVariableOp<feed_forward_sub_net_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype025
3feed_forward_sub_net/dense_4/BiasAdd/ReadVariableOp�
$feed_forward_sub_net/dense_4/BiasAddBiasAdd-feed_forward_sub_net/dense_4/MatMul:product:0;feed_forward_sub_net/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$feed_forward_sub_net/dense_4/BiasAdd�
IdentityIdentity-feed_forward_sub_net/dense_4/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identity�
NoOpNoOp=^feed_forward_sub_net/batch_normalization/Cast/ReadVariableOp?^feed_forward_sub_net/batch_normalization/Cast_1/ReadVariableOp?^feed_forward_sub_net/batch_normalization/Cast_2/ReadVariableOp?^feed_forward_sub_net/batch_normalization/Cast_3/ReadVariableOp?^feed_forward_sub_net/batch_normalization_1/Cast/ReadVariableOpA^feed_forward_sub_net/batch_normalization_1/Cast_1/ReadVariableOpA^feed_forward_sub_net/batch_normalization_1/Cast_2/ReadVariableOpA^feed_forward_sub_net/batch_normalization_1/Cast_3/ReadVariableOp?^feed_forward_sub_net/batch_normalization_2/Cast/ReadVariableOpA^feed_forward_sub_net/batch_normalization_2/Cast_1/ReadVariableOpA^feed_forward_sub_net/batch_normalization_2/Cast_2/ReadVariableOpA^feed_forward_sub_net/batch_normalization_2/Cast_3/ReadVariableOp?^feed_forward_sub_net/batch_normalization_3/Cast/ReadVariableOpA^feed_forward_sub_net/batch_normalization_3/Cast_1/ReadVariableOpA^feed_forward_sub_net/batch_normalization_3/Cast_2/ReadVariableOpA^feed_forward_sub_net/batch_normalization_3/Cast_3/ReadVariableOp?^feed_forward_sub_net/batch_normalization_4/Cast/ReadVariableOpA^feed_forward_sub_net/batch_normalization_4/Cast_1/ReadVariableOpA^feed_forward_sub_net/batch_normalization_4/Cast_2/ReadVariableOpA^feed_forward_sub_net/batch_normalization_4/Cast_3/ReadVariableOp1^feed_forward_sub_net/dense/MatMul/ReadVariableOp3^feed_forward_sub_net/dense_1/MatMul/ReadVariableOp3^feed_forward_sub_net/dense_2/MatMul/ReadVariableOp3^feed_forward_sub_net/dense_3/MatMul/ReadVariableOp4^feed_forward_sub_net/dense_4/BiasAdd/ReadVariableOp3^feed_forward_sub_net/dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:����������: : : : : : : : : : : : : : : : : : : : : : : : : : 2|
<feed_forward_sub_net/batch_normalization/Cast/ReadVariableOp<feed_forward_sub_net/batch_normalization/Cast/ReadVariableOp2�
>feed_forward_sub_net/batch_normalization/Cast_1/ReadVariableOp>feed_forward_sub_net/batch_normalization/Cast_1/ReadVariableOp2�
>feed_forward_sub_net/batch_normalization/Cast_2/ReadVariableOp>feed_forward_sub_net/batch_normalization/Cast_2/ReadVariableOp2�
>feed_forward_sub_net/batch_normalization/Cast_3/ReadVariableOp>feed_forward_sub_net/batch_normalization/Cast_3/ReadVariableOp2�
>feed_forward_sub_net/batch_normalization_1/Cast/ReadVariableOp>feed_forward_sub_net/batch_normalization_1/Cast/ReadVariableOp2�
@feed_forward_sub_net/batch_normalization_1/Cast_1/ReadVariableOp@feed_forward_sub_net/batch_normalization_1/Cast_1/ReadVariableOp2�
@feed_forward_sub_net/batch_normalization_1/Cast_2/ReadVariableOp@feed_forward_sub_net/batch_normalization_1/Cast_2/ReadVariableOp2�
@feed_forward_sub_net/batch_normalization_1/Cast_3/ReadVariableOp@feed_forward_sub_net/batch_normalization_1/Cast_3/ReadVariableOp2�
>feed_forward_sub_net/batch_normalization_2/Cast/ReadVariableOp>feed_forward_sub_net/batch_normalization_2/Cast/ReadVariableOp2�
@feed_forward_sub_net/batch_normalization_2/Cast_1/ReadVariableOp@feed_forward_sub_net/batch_normalization_2/Cast_1/ReadVariableOp2�
@feed_forward_sub_net/batch_normalization_2/Cast_2/ReadVariableOp@feed_forward_sub_net/batch_normalization_2/Cast_2/ReadVariableOp2�
@feed_forward_sub_net/batch_normalization_2/Cast_3/ReadVariableOp@feed_forward_sub_net/batch_normalization_2/Cast_3/ReadVariableOp2�
>feed_forward_sub_net/batch_normalization_3/Cast/ReadVariableOp>feed_forward_sub_net/batch_normalization_3/Cast/ReadVariableOp2�
@feed_forward_sub_net/batch_normalization_3/Cast_1/ReadVariableOp@feed_forward_sub_net/batch_normalization_3/Cast_1/ReadVariableOp2�
@feed_forward_sub_net/batch_normalization_3/Cast_2/ReadVariableOp@feed_forward_sub_net/batch_normalization_3/Cast_2/ReadVariableOp2�
@feed_forward_sub_net/batch_normalization_3/Cast_3/ReadVariableOp@feed_forward_sub_net/batch_normalization_3/Cast_3/ReadVariableOp2�
>feed_forward_sub_net/batch_normalization_4/Cast/ReadVariableOp>feed_forward_sub_net/batch_normalization_4/Cast/ReadVariableOp2�
@feed_forward_sub_net/batch_normalization_4/Cast_1/ReadVariableOp@feed_forward_sub_net/batch_normalization_4/Cast_1/ReadVariableOp2�
@feed_forward_sub_net/batch_normalization_4/Cast_2/ReadVariableOp@feed_forward_sub_net/batch_normalization_4/Cast_2/ReadVariableOp2�
@feed_forward_sub_net/batch_normalization_4/Cast_3/ReadVariableOp@feed_forward_sub_net/batch_normalization_4/Cast_3/ReadVariableOp2d
0feed_forward_sub_net/dense/MatMul/ReadVariableOp0feed_forward_sub_net/dense/MatMul/ReadVariableOp2h
2feed_forward_sub_net/dense_1/MatMul/ReadVariableOp2feed_forward_sub_net/dense_1/MatMul/ReadVariableOp2h
2feed_forward_sub_net/dense_2/MatMul/ReadVariableOp2feed_forward_sub_net/dense_2/MatMul/ReadVariableOp2h
2feed_forward_sub_net/dense_3/MatMul/ReadVariableOp2feed_forward_sub_net/dense_3/MatMul/ReadVariableOp2j
3feed_forward_sub_net/dense_4/BiasAdd/ReadVariableOp3feed_forward_sub_net/dense_4/BiasAdd/ReadVariableOp2h
2feed_forward_sub_net/dense_4/MatMul/ReadVariableOp2feed_forward_sub_net/dense_4/MatMul/ReadVariableOp:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�

�
C__inference_dense_4_layer_call_and_return_conditional_losses_309541

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddl
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_309085

inputs+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�-
cast_2_readvariableop_resource:	�-
cast_3_readvariableop_resource:	�
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOp�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_1/ReadVariableOp�
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_2/ReadVariableOp�
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_3/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:����������2

Identity�
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
C__inference_dense_3_layer_call_and_return_conditional_losses_309524

inputs2
matmul_readvariableop_resource:
��
identity��MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMull
IdentityIdentityMatMul:product:0^NoOp*
T0*(
_output_shapes
:����������2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:����������: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
5__inference_feed_forward_sub_net_layer_call_fn_309065
input_1
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�
	unknown_7:	�
	unknown_8:
��
	unknown_9:	�

unknown_10:	�

unknown_11:	�

unknown_12:	�

unknown_13:
��

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:	�

unknown_18:
��

unknown_19:	�

unknown_20:	�

unknown_21:	�

unknown_22:	�

unknown_23:
��

unknown_24:	�
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
 *(
_output_shapes
:����������*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_feed_forward_sub_net_layer_call_and_return_conditional_losses_3079422
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:����������: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�
�
A__inference_dense_layer_call_and_return_conditional_losses_307622

inputs2
matmul_readvariableop_resource:
��
identity��MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMull
IdentityIdentityMatMul:product:0^NoOp*
T0*(
_output_shapes
:����������2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:����������: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�,
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_309121

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�
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
:	�*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	�2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������2
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
:	�*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
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
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes	
:�2
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
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_1/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:����������2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_3_layer_call_fn_309380

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3072912
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�,
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_309203

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�
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
:	�*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	�2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������2
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
:	�*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
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
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes	
:�2
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
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_1/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:����������2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
P__inference_feed_forward_sub_net_layer_call_and_return_conditional_losses_308545
xJ
;batch_normalization_assignmovingavg_readvariableop_resource:	�L
=batch_normalization_assignmovingavg_1_readvariableop_resource:	�?
0batch_normalization_cast_readvariableop_resource:	�A
2batch_normalization_cast_1_readvariableop_resource:	�8
$dense_matmul_readvariableop_resource:
��L
=batch_normalization_1_assignmovingavg_readvariableop_resource:	�N
?batch_normalization_1_assignmovingavg_1_readvariableop_resource:	�A
2batch_normalization_1_cast_readvariableop_resource:	�C
4batch_normalization_1_cast_1_readvariableop_resource:	�:
&dense_1_matmul_readvariableop_resource:
��L
=batch_normalization_2_assignmovingavg_readvariableop_resource:	�N
?batch_normalization_2_assignmovingavg_1_readvariableop_resource:	�A
2batch_normalization_2_cast_readvariableop_resource:	�C
4batch_normalization_2_cast_1_readvariableop_resource:	�:
&dense_2_matmul_readvariableop_resource:
��L
=batch_normalization_3_assignmovingavg_readvariableop_resource:	�N
?batch_normalization_3_assignmovingavg_1_readvariableop_resource:	�A
2batch_normalization_3_cast_readvariableop_resource:	�C
4batch_normalization_3_cast_1_readvariableop_resource:	�:
&dense_3_matmul_readvariableop_resource:
��L
=batch_normalization_4_assignmovingavg_readvariableop_resource:	�N
?batch_normalization_4_assignmovingavg_1_readvariableop_resource:	�A
2batch_normalization_4_cast_readvariableop_resource:	�C
4batch_normalization_4_cast_1_readvariableop_resource:	�:
&dense_4_matmul_readvariableop_resource:
��6
'dense_4_biasadd_readvariableop_resource:	�
identity��#batch_normalization/AssignMovingAvg�2batch_normalization/AssignMovingAvg/ReadVariableOp�%batch_normalization/AssignMovingAvg_1�4batch_normalization/AssignMovingAvg_1/ReadVariableOp�'batch_normalization/Cast/ReadVariableOp�)batch_normalization/Cast_1/ReadVariableOp�%batch_normalization_1/AssignMovingAvg�4batch_normalization_1/AssignMovingAvg/ReadVariableOp�'batch_normalization_1/AssignMovingAvg_1�6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_1/Cast/ReadVariableOp�+batch_normalization_1/Cast_1/ReadVariableOp�%batch_normalization_2/AssignMovingAvg�4batch_normalization_2/AssignMovingAvg/ReadVariableOp�'batch_normalization_2/AssignMovingAvg_1�6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_2/Cast/ReadVariableOp�+batch_normalization_2/Cast_1/ReadVariableOp�%batch_normalization_3/AssignMovingAvg�4batch_normalization_3/AssignMovingAvg/ReadVariableOp�'batch_normalization_3/AssignMovingAvg_1�6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_3/Cast/ReadVariableOp�+batch_normalization_3/Cast_1/ReadVariableOp�%batch_normalization_4/AssignMovingAvg�4batch_normalization_4/AssignMovingAvg/ReadVariableOp�'batch_normalization_4/AssignMovingAvg_1�6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_4/Cast/ReadVariableOp�+batch_normalization_4/Cast_1/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/MatMul/ReadVariableOp�dense_3/MatMul/ReadVariableOp�dense_4/BiasAdd/ReadVariableOp�dense_4/MatMul/ReadVariableOp�
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 24
2batch_normalization/moments/mean/reduction_indices�
 batch_normalization/moments/meanMeanx;batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2"
 batch_normalization/moments/mean�
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:	�2*
(batch_normalization/moments/StopGradient�
-batch_normalization/moments/SquaredDifferenceSquaredDifferencex1batch_normalization/moments/StopGradient:output:0*
T0*(
_output_shapes
:����������2/
-batch_normalization/moments/SquaredDifference�
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 28
6batch_normalization/moments/variance/reduction_indices�
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2&
$batch_normalization/moments/variance�
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2%
#batch_normalization/moments/Squeeze�
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2'
%batch_normalization/moments/Squeeze_1�
)batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2+
)batch_normalization/AssignMovingAvg/decay�
(batch_normalization/AssignMovingAvg/CastCast2batch_normalization/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2*
(batch_normalization/AssignMovingAvg/Cast�
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp;batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype024
2batch_normalization/AssignMovingAvg/ReadVariableOp�
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes	
:�2)
'batch_normalization/AssignMovingAvg/sub�
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:0,batch_normalization/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes	
:�2)
'batch_normalization/AssignMovingAvg/mul�
#batch_normalization/AssignMovingAvgAssignSubVariableOp;batch_normalization_assignmovingavg_readvariableop_resource+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02%
#batch_normalization/AssignMovingAvg�
+batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization/AssignMovingAvg_1/decay�
*batch_normalization/AssignMovingAvg_1/CastCast4batch_normalization/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2,
*batch_normalization/AssignMovingAvg_1/Cast�
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOp�
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�2+
)batch_normalization/AssignMovingAvg_1/sub�
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:0.batch_normalization/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes	
:�2+
)batch_normalization/AssignMovingAvg_1/mul�
%batch_normalization/AssignMovingAvg_1AssignSubVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization/AssignMovingAvg_1�
'batch_normalization/Cast/ReadVariableOpReadVariableOp0batch_normalization_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02)
'batch_normalization/Cast/ReadVariableOp�
)batch_normalization/Cast_1/ReadVariableOpReadVariableOp2batch_normalization_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization/Cast_1/ReadVariableOp�
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2%
#batch_normalization/batchnorm/add/y�
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2#
!batch_normalization/batchnorm/add�
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization/batchnorm/Rsqrt�
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:01batch_normalization/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2#
!batch_normalization/batchnorm/mul�
#batch_normalization/batchnorm/mul_1Mulx%batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2%
#batch_normalization/batchnorm/mul_1�
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization/batchnorm/mul_2�
!batch_normalization/batchnorm/subSub/batch_normalization/Cast/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2#
!batch_normalization/batchnorm/sub�
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2%
#batch_normalization/batchnorm/add_1�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMul'batch_normalization/batchnorm/add_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/MatMul�
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_1/moments/mean/reduction_indices�
"batch_normalization_1/moments/meanMeandense/MatMul:product:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2$
"batch_normalization_1/moments/mean�
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:	�2,
*batch_normalization_1/moments/StopGradient�
/batch_normalization_1/moments/SquaredDifferenceSquaredDifferencedense/MatMul:product:03batch_normalization_1/moments/StopGradient:output:0*
T0*(
_output_shapes
:����������21
/batch_normalization_1/moments/SquaredDifference�
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_1/moments/variance/reduction_indices�
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2(
&batch_normalization_1/moments/variance�
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2'
%batch_normalization_1/moments/Squeeze�
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2)
'batch_normalization_1/moments/Squeeze_1�
+batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization_1/AssignMovingAvg/decay�
*batch_normalization_1/AssignMovingAvg/CastCast4batch_normalization_1/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2,
*batch_normalization_1/AssignMovingAvg/Cast�
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype026
4batch_normalization_1/AssignMovingAvg/ReadVariableOp�
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes	
:�2+
)batch_normalization_1/AssignMovingAvg/sub�
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:0.batch_normalization_1/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes	
:�2+
)batch_normalization_1/AssignMovingAvg/mul�
%batch_normalization_1/AssignMovingAvgAssignSubVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_1/AssignMovingAvg�
-batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_1/AssignMovingAvg_1/decay�
,batch_normalization_1/AssignMovingAvg_1/CastCast6batch_normalization_1/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2.
,batch_normalization_1/AssignMovingAvg_1/Cast�
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�2-
+batch_normalization_1/AssignMovingAvg_1/sub�
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:00batch_normalization_1/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes	
:�2-
+batch_normalization_1/AssignMovingAvg_1/mul�
'batch_normalization_1/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_1/AssignMovingAvg_1�
)batch_normalization_1/Cast/ReadVariableOpReadVariableOp2batch_normalization_1_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_1/Cast/ReadVariableOp�
+batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_1/Cast_1/ReadVariableOp�
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2'
%batch_normalization_1/batchnorm/add/y�
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_1/batchnorm/add�
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_1/batchnorm/Rsqrt�
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:03batch_normalization_1/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_1/batchnorm/mul�
%batch_normalization_1/batchnorm/mul_1Muldense/MatMul:product:0'batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_1/batchnorm/mul_1�
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_1/batchnorm/mul_2�
#batch_normalization_1/batchnorm/subSub1batch_normalization_1/Cast/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_1/batchnorm/sub�
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_1/batchnorm/add_1r
ReluRelu)batch_normalization_1/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Relu�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMulRelu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_1/MatMul�
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_2/moments/mean/reduction_indices�
"batch_normalization_2/moments/meanMeandense_1/MatMul:product:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2$
"batch_normalization_2/moments/mean�
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:	�2,
*batch_normalization_2/moments/StopGradient�
/batch_normalization_2/moments/SquaredDifferenceSquaredDifferencedense_1/MatMul:product:03batch_normalization_2/moments/StopGradient:output:0*
T0*(
_output_shapes
:����������21
/batch_normalization_2/moments/SquaredDifference�
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_2/moments/variance/reduction_indices�
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2(
&batch_normalization_2/moments/variance�
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2'
%batch_normalization_2/moments/Squeeze�
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2)
'batch_normalization_2/moments/Squeeze_1�
+batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization_2/AssignMovingAvg/decay�
*batch_normalization_2/AssignMovingAvg/CastCast4batch_normalization_2/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2,
*batch_normalization_2/AssignMovingAvg/Cast�
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype026
4batch_normalization_2/AssignMovingAvg/ReadVariableOp�
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes	
:�2+
)batch_normalization_2/AssignMovingAvg/sub�
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:0.batch_normalization_2/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes	
:�2+
)batch_normalization_2/AssignMovingAvg/mul�
%batch_normalization_2/AssignMovingAvgAssignSubVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_2/AssignMovingAvg�
-batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_2/AssignMovingAvg_1/decay�
,batch_normalization_2/AssignMovingAvg_1/CastCast6batch_normalization_2/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2.
,batch_normalization_2/AssignMovingAvg_1/Cast�
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�2-
+batch_normalization_2/AssignMovingAvg_1/sub�
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:00batch_normalization_2/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes	
:�2-
+batch_normalization_2/AssignMovingAvg_1/mul�
'batch_normalization_2/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_2/AssignMovingAvg_1�
)batch_normalization_2/Cast/ReadVariableOpReadVariableOp2batch_normalization_2_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_2/Cast/ReadVariableOp�
+batch_normalization_2/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_2/Cast_1/ReadVariableOp�
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2'
%batch_normalization_2/batchnorm/add/y�
#batch_normalization_2/batchnorm/addAddV20batch_normalization_2/moments/Squeeze_1:output:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_2/batchnorm/add�
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_2/batchnorm/Rsqrt�
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:03batch_normalization_2/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_2/batchnorm/mul�
%batch_normalization_2/batchnorm/mul_1Muldense_1/MatMul:product:0'batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_2/batchnorm/mul_1�
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_2/batchnorm/mul_2�
#batch_normalization_2/batchnorm/subSub1batch_normalization_2/Cast/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_2/batchnorm/sub�
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_2/batchnorm/add_1v
Relu_1Relu)batch_normalization_2/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Relu_1�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_2/MatMul/ReadVariableOp�
dense_2/MatMulMatMulRelu_1:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_2/MatMul�
4batch_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_3/moments/mean/reduction_indices�
"batch_normalization_3/moments/meanMeandense_2/MatMul:product:0=batch_normalization_3/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2$
"batch_normalization_3/moments/mean�
*batch_normalization_3/moments/StopGradientStopGradient+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes
:	�2,
*batch_normalization_3/moments/StopGradient�
/batch_normalization_3/moments/SquaredDifferenceSquaredDifferencedense_2/MatMul:product:03batch_normalization_3/moments/StopGradient:output:0*
T0*(
_output_shapes
:����������21
/batch_normalization_3/moments/SquaredDifference�
8batch_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_3/moments/variance/reduction_indices�
&batch_normalization_3/moments/varianceMean3batch_normalization_3/moments/SquaredDifference:z:0Abatch_normalization_3/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2(
&batch_normalization_3/moments/variance�
%batch_normalization_3/moments/SqueezeSqueeze+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2'
%batch_normalization_3/moments/Squeeze�
'batch_normalization_3/moments/Squeeze_1Squeeze/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2)
'batch_normalization_3/moments/Squeeze_1�
+batch_normalization_3/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization_3/AssignMovingAvg/decay�
*batch_normalization_3/AssignMovingAvg/CastCast4batch_normalization_3/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2,
*batch_normalization_3/AssignMovingAvg/Cast�
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_3_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype026
4batch_normalization_3/AssignMovingAvg/ReadVariableOp�
)batch_normalization_3/AssignMovingAvg/subSub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_3/moments/Squeeze:output:0*
T0*
_output_shapes	
:�2+
)batch_normalization_3/AssignMovingAvg/sub�
)batch_normalization_3/AssignMovingAvg/mulMul-batch_normalization_3/AssignMovingAvg/sub:z:0.batch_normalization_3/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes	
:�2+
)batch_normalization_3/AssignMovingAvg/mul�
%batch_normalization_3/AssignMovingAvgAssignSubVariableOp=batch_normalization_3_assignmovingavg_readvariableop_resource-batch_normalization_3/AssignMovingAvg/mul:z:05^batch_normalization_3/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_3/AssignMovingAvg�
-batch_normalization_3/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_3/AssignMovingAvg_1/decay�
,batch_normalization_3/AssignMovingAvg_1/CastCast6batch_normalization_3/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2.
,batch_normalization_3/AssignMovingAvg_1/Cast�
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_3_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_3/AssignMovingAvg_1/subSub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_3/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�2-
+batch_normalization_3/AssignMovingAvg_1/sub�
+batch_normalization_3/AssignMovingAvg_1/mulMul/batch_normalization_3/AssignMovingAvg_1/sub:z:00batch_normalization_3/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes	
:�2-
+batch_normalization_3/AssignMovingAvg_1/mul�
'batch_normalization_3/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_3_assignmovingavg_1_readvariableop_resource/batch_normalization_3/AssignMovingAvg_1/mul:z:07^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_3/AssignMovingAvg_1�
)batch_normalization_3/Cast/ReadVariableOpReadVariableOp2batch_normalization_3_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_3/Cast/ReadVariableOp�
+batch_normalization_3/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_3_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_3/Cast_1/ReadVariableOp�
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2'
%batch_normalization_3/batchnorm/add/y�
#batch_normalization_3/batchnorm/addAddV20batch_normalization_3/moments/Squeeze_1:output:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_3/batchnorm/add�
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_3/batchnorm/Rsqrt�
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:03batch_normalization_3/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_3/batchnorm/mul�
%batch_normalization_3/batchnorm/mul_1Muldense_2/MatMul:product:0'batch_normalization_3/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_3/batchnorm/mul_1�
%batch_normalization_3/batchnorm/mul_2Mul.batch_normalization_3/moments/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_3/batchnorm/mul_2�
#batch_normalization_3/batchnorm/subSub1batch_normalization_3/Cast/ReadVariableOp:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_3/batchnorm/sub�
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_3/batchnorm/add_1v
Relu_2Relu)batch_normalization_3/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Relu_2�
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_3/MatMul/ReadVariableOp�
dense_3/MatMulMatMulRelu_2:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_3/MatMul�
4batch_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_4/moments/mean/reduction_indices�
"batch_normalization_4/moments/meanMeandense_3/MatMul:product:0=batch_normalization_4/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2$
"batch_normalization_4/moments/mean�
*batch_normalization_4/moments/StopGradientStopGradient+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes
:	�2,
*batch_normalization_4/moments/StopGradient�
/batch_normalization_4/moments/SquaredDifferenceSquaredDifferencedense_3/MatMul:product:03batch_normalization_4/moments/StopGradient:output:0*
T0*(
_output_shapes
:����������21
/batch_normalization_4/moments/SquaredDifference�
8batch_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_4/moments/variance/reduction_indices�
&batch_normalization_4/moments/varianceMean3batch_normalization_4/moments/SquaredDifference:z:0Abatch_normalization_4/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2(
&batch_normalization_4/moments/variance�
%batch_normalization_4/moments/SqueezeSqueeze+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2'
%batch_normalization_4/moments/Squeeze�
'batch_normalization_4/moments/Squeeze_1Squeeze/batch_normalization_4/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2)
'batch_normalization_4/moments/Squeeze_1�
+batch_normalization_4/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization_4/AssignMovingAvg/decay�
*batch_normalization_4/AssignMovingAvg/CastCast4batch_normalization_4/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2,
*batch_normalization_4/AssignMovingAvg/Cast�
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_4_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype026
4batch_normalization_4/AssignMovingAvg/ReadVariableOp�
)batch_normalization_4/AssignMovingAvg/subSub<batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_4/moments/Squeeze:output:0*
T0*
_output_shapes	
:�2+
)batch_normalization_4/AssignMovingAvg/sub�
)batch_normalization_4/AssignMovingAvg/mulMul-batch_normalization_4/AssignMovingAvg/sub:z:0.batch_normalization_4/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes	
:�2+
)batch_normalization_4/AssignMovingAvg/mul�
%batch_normalization_4/AssignMovingAvgAssignSubVariableOp=batch_normalization_4_assignmovingavg_readvariableop_resource-batch_normalization_4/AssignMovingAvg/mul:z:05^batch_normalization_4/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_4/AssignMovingAvg�
-batch_normalization_4/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_4/AssignMovingAvg_1/decay�
,batch_normalization_4/AssignMovingAvg_1/CastCast6batch_normalization_4/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2.
,batch_normalization_4/AssignMovingAvg_1/Cast�
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_4_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_4/AssignMovingAvg_1/subSub>batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_4/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�2-
+batch_normalization_4/AssignMovingAvg_1/sub�
+batch_normalization_4/AssignMovingAvg_1/mulMul/batch_normalization_4/AssignMovingAvg_1/sub:z:00batch_normalization_4/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes	
:�2-
+batch_normalization_4/AssignMovingAvg_1/mul�
'batch_normalization_4/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_4_assignmovingavg_1_readvariableop_resource/batch_normalization_4/AssignMovingAvg_1/mul:z:07^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_4/AssignMovingAvg_1�
)batch_normalization_4/Cast/ReadVariableOpReadVariableOp2batch_normalization_4_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_4/Cast/ReadVariableOp�
+batch_normalization_4/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_4_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_4/Cast_1/ReadVariableOp�
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2'
%batch_normalization_4/batchnorm/add/y�
#batch_normalization_4/batchnorm/addAddV20batch_normalization_4/moments/Squeeze_1:output:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_4/batchnorm/add�
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_4/batchnorm/Rsqrt�
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:03batch_normalization_4/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_4/batchnorm/mul�
%batch_normalization_4/batchnorm/mul_1Muldense_3/MatMul:product:0'batch_normalization_4/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_4/batchnorm/mul_1�
%batch_normalization_4/batchnorm/mul_2Mul.batch_normalization_4/moments/Squeeze:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_4/batchnorm/mul_2�
#batch_normalization_4/batchnorm/subSub1batch_normalization_4/Cast/ReadVariableOp:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_4/batchnorm/sub�
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_4/batchnorm/add_1v
Relu_3Relu)batch_normalization_4/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Relu_3�
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_4/MatMul/ReadVariableOp�
dense_4/MatMulMatMulRelu_3:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_4/MatMul�
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_4/BiasAdd/ReadVariableOp�
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_4/BiasAddt
IdentityIdentitydense_4/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identity�
NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp(^batch_normalization/Cast/ReadVariableOp*^batch_normalization/Cast_1/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_1/Cast/ReadVariableOp,^batch_normalization_1/Cast_1/ReadVariableOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_2/Cast/ReadVariableOp,^batch_normalization_2/Cast_1/ReadVariableOp&^batch_normalization_3/AssignMovingAvg5^batch_normalization_3/AssignMovingAvg/ReadVariableOp(^batch_normalization_3/AssignMovingAvg_17^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_3/Cast/ReadVariableOp,^batch_normalization_3/Cast_1/ReadVariableOp&^batch_normalization_4/AssignMovingAvg5^batch_normalization_4/AssignMovingAvg/ReadVariableOp(^batch_normalization_4/AssignMovingAvg_17^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_4/Cast/ReadVariableOp,^batch_normalization_4/Cast_1/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:����������: : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#batch_normalization/AssignMovingAvg#batch_normalization/AssignMovingAvg2h
2batch_normalization/AssignMovingAvg/ReadVariableOp2batch_normalization/AssignMovingAvg/ReadVariableOp2N
%batch_normalization/AssignMovingAvg_1%batch_normalization/AssignMovingAvg_12l
4batch_normalization/AssignMovingAvg_1/ReadVariableOp4batch_normalization/AssignMovingAvg_1/ReadVariableOp2R
'batch_normalization/Cast/ReadVariableOp'batch_normalization/Cast/ReadVariableOp2V
)batch_normalization/Cast_1/ReadVariableOp)batch_normalization/Cast_1/ReadVariableOp2N
%batch_normalization_1/AssignMovingAvg%batch_normalization_1/AssignMovingAvg2l
4batch_normalization_1/AssignMovingAvg/ReadVariableOp4batch_normalization_1/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_1/AssignMovingAvg_1'batch_normalization_1/AssignMovingAvg_12p
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_1/Cast/ReadVariableOp)batch_normalization_1/Cast/ReadVariableOp2Z
+batch_normalization_1/Cast_1/ReadVariableOp+batch_normalization_1/Cast_1/ReadVariableOp2N
%batch_normalization_2/AssignMovingAvg%batch_normalization_2/AssignMovingAvg2l
4batch_normalization_2/AssignMovingAvg/ReadVariableOp4batch_normalization_2/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_2/AssignMovingAvg_1'batch_normalization_2/AssignMovingAvg_12p
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_2/Cast/ReadVariableOp)batch_normalization_2/Cast/ReadVariableOp2Z
+batch_normalization_2/Cast_1/ReadVariableOp+batch_normalization_2/Cast_1/ReadVariableOp2N
%batch_normalization_3/AssignMovingAvg%batch_normalization_3/AssignMovingAvg2l
4batch_normalization_3/AssignMovingAvg/ReadVariableOp4batch_normalization_3/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_3/AssignMovingAvg_1'batch_normalization_3/AssignMovingAvg_12p
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_3/Cast/ReadVariableOp)batch_normalization_3/Cast/ReadVariableOp2Z
+batch_normalization_3/Cast_1/ReadVariableOp+batch_normalization_3/Cast_1/ReadVariableOp2N
%batch_normalization_4/AssignMovingAvg%batch_normalization_4/AssignMovingAvg2l
4batch_normalization_4/AssignMovingAvg/ReadVariableOp4batch_normalization_4/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_4/AssignMovingAvg_1'batch_normalization_4/AssignMovingAvg_12p
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_4/Cast/ReadVariableOp)batch_normalization_4/Cast/ReadVariableOp2Z
+batch_normalization_4/Cast_1/ReadVariableOp+batch_normalization_4/Cast_1/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp:K G
(
_output_shapes
:����������

_user_specified_namex
�
�
C__inference_dense_3_layer_call_and_return_conditional_losses_307685

inputs2
matmul_readvariableop_resource:
��
identity��MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMull
IdentityIdentityMatMul:product:0^NoOp*
T0*(
_output_shapes
:����������2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:����������: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_309331

inputs+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�-
cast_2_readvariableop_resource:	�-
cast_3_readvariableop_resource:	�
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOp�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_1/ReadVariableOp�
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_2/ReadVariableOp�
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_3/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:����������2

Identity�
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�,
�
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_307353

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�
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
:	�*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	�2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������2
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
:	�*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
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
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes	
:�2
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
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_1/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:����������2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�F
�
__inference__traced_save_309651
file_prefixM
Isavev2_feed_forward_sub_net_batch_normalization_gamma_read_readvariableopL
Hsavev2_feed_forward_sub_net_batch_normalization_beta_read_readvariableopO
Ksavev2_feed_forward_sub_net_batch_normalization_1_gamma_read_readvariableopN
Jsavev2_feed_forward_sub_net_batch_normalization_1_beta_read_readvariableopO
Ksavev2_feed_forward_sub_net_batch_normalization_2_gamma_read_readvariableopN
Jsavev2_feed_forward_sub_net_batch_normalization_2_beta_read_readvariableopO
Ksavev2_feed_forward_sub_net_batch_normalization_3_gamma_read_readvariableopN
Jsavev2_feed_forward_sub_net_batch_normalization_3_beta_read_readvariableopO
Ksavev2_feed_forward_sub_net_batch_normalization_4_gamma_read_readvariableopN
Jsavev2_feed_forward_sub_net_batch_normalization_4_beta_read_readvariableop@
<savev2_feed_forward_sub_net_dense_kernel_read_readvariableopB
>savev2_feed_forward_sub_net_dense_1_kernel_read_readvariableopB
>savev2_feed_forward_sub_net_dense_2_kernel_read_readvariableopB
>savev2_feed_forward_sub_net_dense_3_kernel_read_readvariableopB
>savev2_feed_forward_sub_net_dense_4_kernel_read_readvariableop@
<savev2_feed_forward_sub_net_dense_4_bias_read_readvariableopS
Osavev2_feed_forward_sub_net_batch_normalization_moving_mean_read_readvariableopW
Ssavev2_feed_forward_sub_net_batch_normalization_moving_variance_read_readvariableopU
Qsavev2_feed_forward_sub_net_batch_normalization_1_moving_mean_read_readvariableopY
Usavev2_feed_forward_sub_net_batch_normalization_1_moving_variance_read_readvariableopU
Qsavev2_feed_forward_sub_net_batch_normalization_2_moving_mean_read_readvariableopY
Usavev2_feed_forward_sub_net_batch_normalization_2_moving_variance_read_readvariableopU
Qsavev2_feed_forward_sub_net_batch_normalization_3_moving_mean_read_readvariableopY
Usavev2_feed_forward_sub_net_batch_normalization_3_moving_variance_read_readvariableopU
Qsavev2_feed_forward_sub_net_batch_normalization_4_moving_mean_read_readvariableopY
Usavev2_feed_forward_sub_net_batch_normalization_4_moving_variance_read_readvariableop
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
dtype0*�	
value�	B�	B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Isavev2_feed_forward_sub_net_batch_normalization_gamma_read_readvariableopHsavev2_feed_forward_sub_net_batch_normalization_beta_read_readvariableopKsavev2_feed_forward_sub_net_batch_normalization_1_gamma_read_readvariableopJsavev2_feed_forward_sub_net_batch_normalization_1_beta_read_readvariableopKsavev2_feed_forward_sub_net_batch_normalization_2_gamma_read_readvariableopJsavev2_feed_forward_sub_net_batch_normalization_2_beta_read_readvariableopKsavev2_feed_forward_sub_net_batch_normalization_3_gamma_read_readvariableopJsavev2_feed_forward_sub_net_batch_normalization_3_beta_read_readvariableopKsavev2_feed_forward_sub_net_batch_normalization_4_gamma_read_readvariableopJsavev2_feed_forward_sub_net_batch_normalization_4_beta_read_readvariableop<savev2_feed_forward_sub_net_dense_kernel_read_readvariableop>savev2_feed_forward_sub_net_dense_1_kernel_read_readvariableop>savev2_feed_forward_sub_net_dense_2_kernel_read_readvariableop>savev2_feed_forward_sub_net_dense_3_kernel_read_readvariableop>savev2_feed_forward_sub_net_dense_4_kernel_read_readvariableop<savev2_feed_forward_sub_net_dense_4_bias_read_readvariableopOsavev2_feed_forward_sub_net_batch_normalization_moving_mean_read_readvariableopSsavev2_feed_forward_sub_net_batch_normalization_moving_variance_read_readvariableopQsavev2_feed_forward_sub_net_batch_normalization_1_moving_mean_read_readvariableopUsavev2_feed_forward_sub_net_batch_normalization_1_moving_variance_read_readvariableopQsavev2_feed_forward_sub_net_batch_normalization_2_moving_mean_read_readvariableopUsavev2_feed_forward_sub_net_batch_normalization_2_moving_variance_read_readvariableopQsavev2_feed_forward_sub_net_batch_normalization_3_moving_mean_read_readvariableopUsavev2_feed_forward_sub_net_batch_normalization_3_moving_variance_read_readvariableopQsavev2_feed_forward_sub_net_batch_normalization_4_moving_mean_read_readvariableopUsavev2_feed_forward_sub_net_batch_normalization_4_moving_variance_read_readvariableopsavev2_const"/device:CPU:0*
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
�: :�:�:�:�:�:�:�:�:�:�:
��:
��:
��:
��:
��:�:�:�:�:�:�:�:�:�:�:�: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!	

_output_shapes	
:�:!


_output_shapes	
:�:&"
 
_output_shapes
:
��:&"
 
_output_shapes
:
��:&"
 
_output_shapes
:
��:&"
 
_output_shapes
:
��:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:

_output_shapes
: 
��
�
P__inference_feed_forward_sub_net_layer_call_and_return_conditional_losses_308651
input_1?
0batch_normalization_cast_readvariableop_resource:	�A
2batch_normalization_cast_1_readvariableop_resource:	�A
2batch_normalization_cast_2_readvariableop_resource:	�A
2batch_normalization_cast_3_readvariableop_resource:	�8
$dense_matmul_readvariableop_resource:
��A
2batch_normalization_1_cast_readvariableop_resource:	�C
4batch_normalization_1_cast_1_readvariableop_resource:	�C
4batch_normalization_1_cast_2_readvariableop_resource:	�C
4batch_normalization_1_cast_3_readvariableop_resource:	�:
&dense_1_matmul_readvariableop_resource:
��A
2batch_normalization_2_cast_readvariableop_resource:	�C
4batch_normalization_2_cast_1_readvariableop_resource:	�C
4batch_normalization_2_cast_2_readvariableop_resource:	�C
4batch_normalization_2_cast_3_readvariableop_resource:	�:
&dense_2_matmul_readvariableop_resource:
��A
2batch_normalization_3_cast_readvariableop_resource:	�C
4batch_normalization_3_cast_1_readvariableop_resource:	�C
4batch_normalization_3_cast_2_readvariableop_resource:	�C
4batch_normalization_3_cast_3_readvariableop_resource:	�:
&dense_3_matmul_readvariableop_resource:
��A
2batch_normalization_4_cast_readvariableop_resource:	�C
4batch_normalization_4_cast_1_readvariableop_resource:	�C
4batch_normalization_4_cast_2_readvariableop_resource:	�C
4batch_normalization_4_cast_3_readvariableop_resource:	�:
&dense_4_matmul_readvariableop_resource:
��6
'dense_4_biasadd_readvariableop_resource:	�
identity��'batch_normalization/Cast/ReadVariableOp�)batch_normalization/Cast_1/ReadVariableOp�)batch_normalization/Cast_2/ReadVariableOp�)batch_normalization/Cast_3/ReadVariableOp�)batch_normalization_1/Cast/ReadVariableOp�+batch_normalization_1/Cast_1/ReadVariableOp�+batch_normalization_1/Cast_2/ReadVariableOp�+batch_normalization_1/Cast_3/ReadVariableOp�)batch_normalization_2/Cast/ReadVariableOp�+batch_normalization_2/Cast_1/ReadVariableOp�+batch_normalization_2/Cast_2/ReadVariableOp�+batch_normalization_2/Cast_3/ReadVariableOp�)batch_normalization_3/Cast/ReadVariableOp�+batch_normalization_3/Cast_1/ReadVariableOp�+batch_normalization_3/Cast_2/ReadVariableOp�+batch_normalization_3/Cast_3/ReadVariableOp�)batch_normalization_4/Cast/ReadVariableOp�+batch_normalization_4/Cast_1/ReadVariableOp�+batch_normalization_4/Cast_2/ReadVariableOp�+batch_normalization_4/Cast_3/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/MatMul/ReadVariableOp�dense_3/MatMul/ReadVariableOp�dense_4/BiasAdd/ReadVariableOp�dense_4/MatMul/ReadVariableOp�
'batch_normalization/Cast/ReadVariableOpReadVariableOp0batch_normalization_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02)
'batch_normalization/Cast/ReadVariableOp�
)batch_normalization/Cast_1/ReadVariableOpReadVariableOp2batch_normalization_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization/Cast_1/ReadVariableOp�
)batch_normalization/Cast_2/ReadVariableOpReadVariableOp2batch_normalization_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization/Cast_2/ReadVariableOp�
)batch_normalization/Cast_3/ReadVariableOpReadVariableOp2batch_normalization_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization/Cast_3/ReadVariableOp�
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2%
#batch_normalization/batchnorm/add/y�
!batch_normalization/batchnorm/addAddV21batch_normalization/Cast_1/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2#
!batch_normalization/batchnorm/add�
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization/batchnorm/Rsqrt�
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:01batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2#
!batch_normalization/batchnorm/mul�
#batch_normalization/batchnorm/mul_1Mulinput_1%batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2%
#batch_normalization/batchnorm/mul_1�
#batch_normalization/batchnorm/mul_2Mul/batch_normalization/Cast/ReadVariableOp:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization/batchnorm/mul_2�
!batch_normalization/batchnorm/subSub1batch_normalization/Cast_2/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2#
!batch_normalization/batchnorm/sub�
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2%
#batch_normalization/batchnorm/add_1�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMul'batch_normalization/batchnorm/add_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/MatMul�
)batch_normalization_1/Cast/ReadVariableOpReadVariableOp2batch_normalization_1_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_1/Cast/ReadVariableOp�
+batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_1/Cast_1/ReadVariableOp�
+batch_normalization_1/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_1_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_1/Cast_2/ReadVariableOp�
+batch_normalization_1/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_1_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_1/Cast_3/ReadVariableOp�
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2'
%batch_normalization_1/batchnorm/add/y�
#batch_normalization_1/batchnorm/addAddV23batch_normalization_1/Cast_1/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_1/batchnorm/add�
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_1/batchnorm/Rsqrt�
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:03batch_normalization_1/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_1/batchnorm/mul�
%batch_normalization_1/batchnorm/mul_1Muldense/MatMul:product:0'batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_1/batchnorm/mul_1�
%batch_normalization_1/batchnorm/mul_2Mul1batch_normalization_1/Cast/ReadVariableOp:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_1/batchnorm/mul_2�
#batch_normalization_1/batchnorm/subSub3batch_normalization_1/Cast_2/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_1/batchnorm/sub�
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_1/batchnorm/add_1r
ReluRelu)batch_normalization_1/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Relu�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMulRelu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_1/MatMul�
)batch_normalization_2/Cast/ReadVariableOpReadVariableOp2batch_normalization_2_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_2/Cast/ReadVariableOp�
+batch_normalization_2/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_2/Cast_1/ReadVariableOp�
+batch_normalization_2/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_2_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_2/Cast_2/ReadVariableOp�
+batch_normalization_2/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_2_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_2/Cast_3/ReadVariableOp�
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2'
%batch_normalization_2/batchnorm/add/y�
#batch_normalization_2/batchnorm/addAddV23batch_normalization_2/Cast_1/ReadVariableOp:value:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_2/batchnorm/add�
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_2/batchnorm/Rsqrt�
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:03batch_normalization_2/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_2/batchnorm/mul�
%batch_normalization_2/batchnorm/mul_1Muldense_1/MatMul:product:0'batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_2/batchnorm/mul_1�
%batch_normalization_2/batchnorm/mul_2Mul1batch_normalization_2/Cast/ReadVariableOp:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_2/batchnorm/mul_2�
#batch_normalization_2/batchnorm/subSub3batch_normalization_2/Cast_2/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_2/batchnorm/sub�
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_2/batchnorm/add_1v
Relu_1Relu)batch_normalization_2/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Relu_1�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_2/MatMul/ReadVariableOp�
dense_2/MatMulMatMulRelu_1:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_2/MatMul�
)batch_normalization_3/Cast/ReadVariableOpReadVariableOp2batch_normalization_3_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_3/Cast/ReadVariableOp�
+batch_normalization_3/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_3_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_3/Cast_1/ReadVariableOp�
+batch_normalization_3/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_3_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_3/Cast_2/ReadVariableOp�
+batch_normalization_3/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_3_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_3/Cast_3/ReadVariableOp�
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2'
%batch_normalization_3/batchnorm/add/y�
#batch_normalization_3/batchnorm/addAddV23batch_normalization_3/Cast_1/ReadVariableOp:value:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_3/batchnorm/add�
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_3/batchnorm/Rsqrt�
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:03batch_normalization_3/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_3/batchnorm/mul�
%batch_normalization_3/batchnorm/mul_1Muldense_2/MatMul:product:0'batch_normalization_3/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_3/batchnorm/mul_1�
%batch_normalization_3/batchnorm/mul_2Mul1batch_normalization_3/Cast/ReadVariableOp:value:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_3/batchnorm/mul_2�
#batch_normalization_3/batchnorm/subSub3batch_normalization_3/Cast_2/ReadVariableOp:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_3/batchnorm/sub�
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_3/batchnorm/add_1v
Relu_2Relu)batch_normalization_3/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Relu_2�
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_3/MatMul/ReadVariableOp�
dense_3/MatMulMatMulRelu_2:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_3/MatMul�
)batch_normalization_4/Cast/ReadVariableOpReadVariableOp2batch_normalization_4_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_4/Cast/ReadVariableOp�
+batch_normalization_4/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_4_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_4/Cast_1/ReadVariableOp�
+batch_normalization_4/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_4_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_4/Cast_2/ReadVariableOp�
+batch_normalization_4/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_4_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_4/Cast_3/ReadVariableOp�
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2'
%batch_normalization_4/batchnorm/add/y�
#batch_normalization_4/batchnorm/addAddV23batch_normalization_4/Cast_1/ReadVariableOp:value:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_4/batchnorm/add�
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_4/batchnorm/Rsqrt�
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:03batch_normalization_4/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_4/batchnorm/mul�
%batch_normalization_4/batchnorm/mul_1Muldense_3/MatMul:product:0'batch_normalization_4/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_4/batchnorm/mul_1�
%batch_normalization_4/batchnorm/mul_2Mul1batch_normalization_4/Cast/ReadVariableOp:value:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_4/batchnorm/mul_2�
#batch_normalization_4/batchnorm/subSub3batch_normalization_4/Cast_2/ReadVariableOp:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_4/batchnorm/sub�
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_4/batchnorm/add_1v
Relu_3Relu)batch_normalization_4/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Relu_3�
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_4/MatMul/ReadVariableOp�
dense_4/MatMulMatMulRelu_3:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_4/MatMul�
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_4/BiasAdd/ReadVariableOp�
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_4/BiasAddt
IdentityIdentitydense_4/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identity�	
NoOpNoOp(^batch_normalization/Cast/ReadVariableOp*^batch_normalization/Cast_1/ReadVariableOp*^batch_normalization/Cast_2/ReadVariableOp*^batch_normalization/Cast_3/ReadVariableOp*^batch_normalization_1/Cast/ReadVariableOp,^batch_normalization_1/Cast_1/ReadVariableOp,^batch_normalization_1/Cast_2/ReadVariableOp,^batch_normalization_1/Cast_3/ReadVariableOp*^batch_normalization_2/Cast/ReadVariableOp,^batch_normalization_2/Cast_1/ReadVariableOp,^batch_normalization_2/Cast_2/ReadVariableOp,^batch_normalization_2/Cast_3/ReadVariableOp*^batch_normalization_3/Cast/ReadVariableOp,^batch_normalization_3/Cast_1/ReadVariableOp,^batch_normalization_3/Cast_2/ReadVariableOp,^batch_normalization_3/Cast_3/ReadVariableOp*^batch_normalization_4/Cast/ReadVariableOp,^batch_normalization_4/Cast_1/ReadVariableOp,^batch_normalization_4/Cast_2/ReadVariableOp,^batch_normalization_4/Cast_3/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:����������: : : : : : : : : : : : : : : : : : : : : : : : : : 2R
'batch_normalization/Cast/ReadVariableOp'batch_normalization/Cast/ReadVariableOp2V
)batch_normalization/Cast_1/ReadVariableOp)batch_normalization/Cast_1/ReadVariableOp2V
)batch_normalization/Cast_2/ReadVariableOp)batch_normalization/Cast_2/ReadVariableOp2V
)batch_normalization/Cast_3/ReadVariableOp)batch_normalization/Cast_3/ReadVariableOp2V
)batch_normalization_1/Cast/ReadVariableOp)batch_normalization_1/Cast/ReadVariableOp2Z
+batch_normalization_1/Cast_1/ReadVariableOp+batch_normalization_1/Cast_1/ReadVariableOp2Z
+batch_normalization_1/Cast_2/ReadVariableOp+batch_normalization_1/Cast_2/ReadVariableOp2Z
+batch_normalization_1/Cast_3/ReadVariableOp+batch_normalization_1/Cast_3/ReadVariableOp2V
)batch_normalization_2/Cast/ReadVariableOp)batch_normalization_2/Cast/ReadVariableOp2Z
+batch_normalization_2/Cast_1/ReadVariableOp+batch_normalization_2/Cast_1/ReadVariableOp2Z
+batch_normalization_2/Cast_2/ReadVariableOp+batch_normalization_2/Cast_2/ReadVariableOp2Z
+batch_normalization_2/Cast_3/ReadVariableOp+batch_normalization_2/Cast_3/ReadVariableOp2V
)batch_normalization_3/Cast/ReadVariableOp)batch_normalization_3/Cast/ReadVariableOp2Z
+batch_normalization_3/Cast_1/ReadVariableOp+batch_normalization_3/Cast_1/ReadVariableOp2Z
+batch_normalization_3/Cast_2/ReadVariableOp+batch_normalization_3/Cast_2/ReadVariableOp2Z
+batch_normalization_3/Cast_3/ReadVariableOp+batch_normalization_3/Cast_3/ReadVariableOp2V
)batch_normalization_4/Cast/ReadVariableOp)batch_normalization_4/Cast/ReadVariableOp2Z
+batch_normalization_4/Cast_1/ReadVariableOp+batch_normalization_4/Cast_1/ReadVariableOp2Z
+batch_normalization_4/Cast_2/ReadVariableOp+batch_normalization_4/Cast_2/ReadVariableOp2Z
+batch_normalization_4/Cast_3/ReadVariableOp+batch_normalization_4/Cast_3/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�
�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_307457

inputs+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�-
cast_2_readvariableop_resource:	�-
cast_3_readvariableop_resource:	�
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOp�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_1/ReadVariableOp�
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_2/ReadVariableOp�
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_3/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:����������2

Identity�
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_4_layer_call_fn_309475

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_3075192
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
P__inference_feed_forward_sub_net_layer_call_and_return_conditional_losses_308837
input_1J
;batch_normalization_assignmovingavg_readvariableop_resource:	�L
=batch_normalization_assignmovingavg_1_readvariableop_resource:	�?
0batch_normalization_cast_readvariableop_resource:	�A
2batch_normalization_cast_1_readvariableop_resource:	�8
$dense_matmul_readvariableop_resource:
��L
=batch_normalization_1_assignmovingavg_readvariableop_resource:	�N
?batch_normalization_1_assignmovingavg_1_readvariableop_resource:	�A
2batch_normalization_1_cast_readvariableop_resource:	�C
4batch_normalization_1_cast_1_readvariableop_resource:	�:
&dense_1_matmul_readvariableop_resource:
��L
=batch_normalization_2_assignmovingavg_readvariableop_resource:	�N
?batch_normalization_2_assignmovingavg_1_readvariableop_resource:	�A
2batch_normalization_2_cast_readvariableop_resource:	�C
4batch_normalization_2_cast_1_readvariableop_resource:	�:
&dense_2_matmul_readvariableop_resource:
��L
=batch_normalization_3_assignmovingavg_readvariableop_resource:	�N
?batch_normalization_3_assignmovingavg_1_readvariableop_resource:	�A
2batch_normalization_3_cast_readvariableop_resource:	�C
4batch_normalization_3_cast_1_readvariableop_resource:	�:
&dense_3_matmul_readvariableop_resource:
��L
=batch_normalization_4_assignmovingavg_readvariableop_resource:	�N
?batch_normalization_4_assignmovingavg_1_readvariableop_resource:	�A
2batch_normalization_4_cast_readvariableop_resource:	�C
4batch_normalization_4_cast_1_readvariableop_resource:	�:
&dense_4_matmul_readvariableop_resource:
��6
'dense_4_biasadd_readvariableop_resource:	�
identity��#batch_normalization/AssignMovingAvg�2batch_normalization/AssignMovingAvg/ReadVariableOp�%batch_normalization/AssignMovingAvg_1�4batch_normalization/AssignMovingAvg_1/ReadVariableOp�'batch_normalization/Cast/ReadVariableOp�)batch_normalization/Cast_1/ReadVariableOp�%batch_normalization_1/AssignMovingAvg�4batch_normalization_1/AssignMovingAvg/ReadVariableOp�'batch_normalization_1/AssignMovingAvg_1�6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_1/Cast/ReadVariableOp�+batch_normalization_1/Cast_1/ReadVariableOp�%batch_normalization_2/AssignMovingAvg�4batch_normalization_2/AssignMovingAvg/ReadVariableOp�'batch_normalization_2/AssignMovingAvg_1�6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_2/Cast/ReadVariableOp�+batch_normalization_2/Cast_1/ReadVariableOp�%batch_normalization_3/AssignMovingAvg�4batch_normalization_3/AssignMovingAvg/ReadVariableOp�'batch_normalization_3/AssignMovingAvg_1�6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_3/Cast/ReadVariableOp�+batch_normalization_3/Cast_1/ReadVariableOp�%batch_normalization_4/AssignMovingAvg�4batch_normalization_4/AssignMovingAvg/ReadVariableOp�'batch_normalization_4/AssignMovingAvg_1�6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_4/Cast/ReadVariableOp�+batch_normalization_4/Cast_1/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/MatMul/ReadVariableOp�dense_3/MatMul/ReadVariableOp�dense_4/BiasAdd/ReadVariableOp�dense_4/MatMul/ReadVariableOp�
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 24
2batch_normalization/moments/mean/reduction_indices�
 batch_normalization/moments/meanMeaninput_1;batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2"
 batch_normalization/moments/mean�
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:	�2*
(batch_normalization/moments/StopGradient�
-batch_normalization/moments/SquaredDifferenceSquaredDifferenceinput_11batch_normalization/moments/StopGradient:output:0*
T0*(
_output_shapes
:����������2/
-batch_normalization/moments/SquaredDifference�
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 28
6batch_normalization/moments/variance/reduction_indices�
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2&
$batch_normalization/moments/variance�
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2%
#batch_normalization/moments/Squeeze�
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2'
%batch_normalization/moments/Squeeze_1�
)batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2+
)batch_normalization/AssignMovingAvg/decay�
(batch_normalization/AssignMovingAvg/CastCast2batch_normalization/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2*
(batch_normalization/AssignMovingAvg/Cast�
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp;batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype024
2batch_normalization/AssignMovingAvg/ReadVariableOp�
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes	
:�2)
'batch_normalization/AssignMovingAvg/sub�
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:0,batch_normalization/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes	
:�2)
'batch_normalization/AssignMovingAvg/mul�
#batch_normalization/AssignMovingAvgAssignSubVariableOp;batch_normalization_assignmovingavg_readvariableop_resource+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02%
#batch_normalization/AssignMovingAvg�
+batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization/AssignMovingAvg_1/decay�
*batch_normalization/AssignMovingAvg_1/CastCast4batch_normalization/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2,
*batch_normalization/AssignMovingAvg_1/Cast�
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOp�
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�2+
)batch_normalization/AssignMovingAvg_1/sub�
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:0.batch_normalization/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes	
:�2+
)batch_normalization/AssignMovingAvg_1/mul�
%batch_normalization/AssignMovingAvg_1AssignSubVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization/AssignMovingAvg_1�
'batch_normalization/Cast/ReadVariableOpReadVariableOp0batch_normalization_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02)
'batch_normalization/Cast/ReadVariableOp�
)batch_normalization/Cast_1/ReadVariableOpReadVariableOp2batch_normalization_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization/Cast_1/ReadVariableOp�
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2%
#batch_normalization/batchnorm/add/y�
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2#
!batch_normalization/batchnorm/add�
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization/batchnorm/Rsqrt�
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:01batch_normalization/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2#
!batch_normalization/batchnorm/mul�
#batch_normalization/batchnorm/mul_1Mulinput_1%batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2%
#batch_normalization/batchnorm/mul_1�
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization/batchnorm/mul_2�
!batch_normalization/batchnorm/subSub/batch_normalization/Cast/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2#
!batch_normalization/batchnorm/sub�
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2%
#batch_normalization/batchnorm/add_1�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMul'batch_normalization/batchnorm/add_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/MatMul�
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_1/moments/mean/reduction_indices�
"batch_normalization_1/moments/meanMeandense/MatMul:product:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2$
"batch_normalization_1/moments/mean�
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:	�2,
*batch_normalization_1/moments/StopGradient�
/batch_normalization_1/moments/SquaredDifferenceSquaredDifferencedense/MatMul:product:03batch_normalization_1/moments/StopGradient:output:0*
T0*(
_output_shapes
:����������21
/batch_normalization_1/moments/SquaredDifference�
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_1/moments/variance/reduction_indices�
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2(
&batch_normalization_1/moments/variance�
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2'
%batch_normalization_1/moments/Squeeze�
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2)
'batch_normalization_1/moments/Squeeze_1�
+batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization_1/AssignMovingAvg/decay�
*batch_normalization_1/AssignMovingAvg/CastCast4batch_normalization_1/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2,
*batch_normalization_1/AssignMovingAvg/Cast�
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype026
4batch_normalization_1/AssignMovingAvg/ReadVariableOp�
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes	
:�2+
)batch_normalization_1/AssignMovingAvg/sub�
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:0.batch_normalization_1/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes	
:�2+
)batch_normalization_1/AssignMovingAvg/mul�
%batch_normalization_1/AssignMovingAvgAssignSubVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_1/AssignMovingAvg�
-batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_1/AssignMovingAvg_1/decay�
,batch_normalization_1/AssignMovingAvg_1/CastCast6batch_normalization_1/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2.
,batch_normalization_1/AssignMovingAvg_1/Cast�
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�2-
+batch_normalization_1/AssignMovingAvg_1/sub�
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:00batch_normalization_1/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes	
:�2-
+batch_normalization_1/AssignMovingAvg_1/mul�
'batch_normalization_1/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_1/AssignMovingAvg_1�
)batch_normalization_1/Cast/ReadVariableOpReadVariableOp2batch_normalization_1_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_1/Cast/ReadVariableOp�
+batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_1/Cast_1/ReadVariableOp�
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2'
%batch_normalization_1/batchnorm/add/y�
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_1/batchnorm/add�
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_1/batchnorm/Rsqrt�
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:03batch_normalization_1/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_1/batchnorm/mul�
%batch_normalization_1/batchnorm/mul_1Muldense/MatMul:product:0'batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_1/batchnorm/mul_1�
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_1/batchnorm/mul_2�
#batch_normalization_1/batchnorm/subSub1batch_normalization_1/Cast/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_1/batchnorm/sub�
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_1/batchnorm/add_1r
ReluRelu)batch_normalization_1/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Relu�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMulRelu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_1/MatMul�
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_2/moments/mean/reduction_indices�
"batch_normalization_2/moments/meanMeandense_1/MatMul:product:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2$
"batch_normalization_2/moments/mean�
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:	�2,
*batch_normalization_2/moments/StopGradient�
/batch_normalization_2/moments/SquaredDifferenceSquaredDifferencedense_1/MatMul:product:03batch_normalization_2/moments/StopGradient:output:0*
T0*(
_output_shapes
:����������21
/batch_normalization_2/moments/SquaredDifference�
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_2/moments/variance/reduction_indices�
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2(
&batch_normalization_2/moments/variance�
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2'
%batch_normalization_2/moments/Squeeze�
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2)
'batch_normalization_2/moments/Squeeze_1�
+batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization_2/AssignMovingAvg/decay�
*batch_normalization_2/AssignMovingAvg/CastCast4batch_normalization_2/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2,
*batch_normalization_2/AssignMovingAvg/Cast�
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype026
4batch_normalization_2/AssignMovingAvg/ReadVariableOp�
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes	
:�2+
)batch_normalization_2/AssignMovingAvg/sub�
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:0.batch_normalization_2/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes	
:�2+
)batch_normalization_2/AssignMovingAvg/mul�
%batch_normalization_2/AssignMovingAvgAssignSubVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_2/AssignMovingAvg�
-batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_2/AssignMovingAvg_1/decay�
,batch_normalization_2/AssignMovingAvg_1/CastCast6batch_normalization_2/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2.
,batch_normalization_2/AssignMovingAvg_1/Cast�
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�2-
+batch_normalization_2/AssignMovingAvg_1/sub�
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:00batch_normalization_2/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes	
:�2-
+batch_normalization_2/AssignMovingAvg_1/mul�
'batch_normalization_2/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_2/AssignMovingAvg_1�
)batch_normalization_2/Cast/ReadVariableOpReadVariableOp2batch_normalization_2_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_2/Cast/ReadVariableOp�
+batch_normalization_2/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_2/Cast_1/ReadVariableOp�
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2'
%batch_normalization_2/batchnorm/add/y�
#batch_normalization_2/batchnorm/addAddV20batch_normalization_2/moments/Squeeze_1:output:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_2/batchnorm/add�
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_2/batchnorm/Rsqrt�
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:03batch_normalization_2/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_2/batchnorm/mul�
%batch_normalization_2/batchnorm/mul_1Muldense_1/MatMul:product:0'batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_2/batchnorm/mul_1�
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_2/batchnorm/mul_2�
#batch_normalization_2/batchnorm/subSub1batch_normalization_2/Cast/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_2/batchnorm/sub�
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_2/batchnorm/add_1v
Relu_1Relu)batch_normalization_2/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Relu_1�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_2/MatMul/ReadVariableOp�
dense_2/MatMulMatMulRelu_1:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_2/MatMul�
4batch_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_3/moments/mean/reduction_indices�
"batch_normalization_3/moments/meanMeandense_2/MatMul:product:0=batch_normalization_3/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2$
"batch_normalization_3/moments/mean�
*batch_normalization_3/moments/StopGradientStopGradient+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes
:	�2,
*batch_normalization_3/moments/StopGradient�
/batch_normalization_3/moments/SquaredDifferenceSquaredDifferencedense_2/MatMul:product:03batch_normalization_3/moments/StopGradient:output:0*
T0*(
_output_shapes
:����������21
/batch_normalization_3/moments/SquaredDifference�
8batch_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_3/moments/variance/reduction_indices�
&batch_normalization_3/moments/varianceMean3batch_normalization_3/moments/SquaredDifference:z:0Abatch_normalization_3/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2(
&batch_normalization_3/moments/variance�
%batch_normalization_3/moments/SqueezeSqueeze+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2'
%batch_normalization_3/moments/Squeeze�
'batch_normalization_3/moments/Squeeze_1Squeeze/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2)
'batch_normalization_3/moments/Squeeze_1�
+batch_normalization_3/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization_3/AssignMovingAvg/decay�
*batch_normalization_3/AssignMovingAvg/CastCast4batch_normalization_3/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2,
*batch_normalization_3/AssignMovingAvg/Cast�
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_3_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype026
4batch_normalization_3/AssignMovingAvg/ReadVariableOp�
)batch_normalization_3/AssignMovingAvg/subSub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_3/moments/Squeeze:output:0*
T0*
_output_shapes	
:�2+
)batch_normalization_3/AssignMovingAvg/sub�
)batch_normalization_3/AssignMovingAvg/mulMul-batch_normalization_3/AssignMovingAvg/sub:z:0.batch_normalization_3/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes	
:�2+
)batch_normalization_3/AssignMovingAvg/mul�
%batch_normalization_3/AssignMovingAvgAssignSubVariableOp=batch_normalization_3_assignmovingavg_readvariableop_resource-batch_normalization_3/AssignMovingAvg/mul:z:05^batch_normalization_3/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_3/AssignMovingAvg�
-batch_normalization_3/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_3/AssignMovingAvg_1/decay�
,batch_normalization_3/AssignMovingAvg_1/CastCast6batch_normalization_3/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2.
,batch_normalization_3/AssignMovingAvg_1/Cast�
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_3_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_3/AssignMovingAvg_1/subSub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_3/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�2-
+batch_normalization_3/AssignMovingAvg_1/sub�
+batch_normalization_3/AssignMovingAvg_1/mulMul/batch_normalization_3/AssignMovingAvg_1/sub:z:00batch_normalization_3/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes	
:�2-
+batch_normalization_3/AssignMovingAvg_1/mul�
'batch_normalization_3/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_3_assignmovingavg_1_readvariableop_resource/batch_normalization_3/AssignMovingAvg_1/mul:z:07^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_3/AssignMovingAvg_1�
)batch_normalization_3/Cast/ReadVariableOpReadVariableOp2batch_normalization_3_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_3/Cast/ReadVariableOp�
+batch_normalization_3/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_3_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_3/Cast_1/ReadVariableOp�
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2'
%batch_normalization_3/batchnorm/add/y�
#batch_normalization_3/batchnorm/addAddV20batch_normalization_3/moments/Squeeze_1:output:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_3/batchnorm/add�
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_3/batchnorm/Rsqrt�
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:03batch_normalization_3/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_3/batchnorm/mul�
%batch_normalization_3/batchnorm/mul_1Muldense_2/MatMul:product:0'batch_normalization_3/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_3/batchnorm/mul_1�
%batch_normalization_3/batchnorm/mul_2Mul.batch_normalization_3/moments/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_3/batchnorm/mul_2�
#batch_normalization_3/batchnorm/subSub1batch_normalization_3/Cast/ReadVariableOp:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_3/batchnorm/sub�
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_3/batchnorm/add_1v
Relu_2Relu)batch_normalization_3/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Relu_2�
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_3/MatMul/ReadVariableOp�
dense_3/MatMulMatMulRelu_2:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_3/MatMul�
4batch_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_4/moments/mean/reduction_indices�
"batch_normalization_4/moments/meanMeandense_3/MatMul:product:0=batch_normalization_4/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2$
"batch_normalization_4/moments/mean�
*batch_normalization_4/moments/StopGradientStopGradient+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes
:	�2,
*batch_normalization_4/moments/StopGradient�
/batch_normalization_4/moments/SquaredDifferenceSquaredDifferencedense_3/MatMul:product:03batch_normalization_4/moments/StopGradient:output:0*
T0*(
_output_shapes
:����������21
/batch_normalization_4/moments/SquaredDifference�
8batch_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_4/moments/variance/reduction_indices�
&batch_normalization_4/moments/varianceMean3batch_normalization_4/moments/SquaredDifference:z:0Abatch_normalization_4/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2(
&batch_normalization_4/moments/variance�
%batch_normalization_4/moments/SqueezeSqueeze+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2'
%batch_normalization_4/moments/Squeeze�
'batch_normalization_4/moments/Squeeze_1Squeeze/batch_normalization_4/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2)
'batch_normalization_4/moments/Squeeze_1�
+batch_normalization_4/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization_4/AssignMovingAvg/decay�
*batch_normalization_4/AssignMovingAvg/CastCast4batch_normalization_4/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2,
*batch_normalization_4/AssignMovingAvg/Cast�
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_4_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype026
4batch_normalization_4/AssignMovingAvg/ReadVariableOp�
)batch_normalization_4/AssignMovingAvg/subSub<batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_4/moments/Squeeze:output:0*
T0*
_output_shapes	
:�2+
)batch_normalization_4/AssignMovingAvg/sub�
)batch_normalization_4/AssignMovingAvg/mulMul-batch_normalization_4/AssignMovingAvg/sub:z:0.batch_normalization_4/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes	
:�2+
)batch_normalization_4/AssignMovingAvg/mul�
%batch_normalization_4/AssignMovingAvgAssignSubVariableOp=batch_normalization_4_assignmovingavg_readvariableop_resource-batch_normalization_4/AssignMovingAvg/mul:z:05^batch_normalization_4/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_4/AssignMovingAvg�
-batch_normalization_4/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_4/AssignMovingAvg_1/decay�
,batch_normalization_4/AssignMovingAvg_1/CastCast6batch_normalization_4/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2.
,batch_normalization_4/AssignMovingAvg_1/Cast�
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_4_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_4/AssignMovingAvg_1/subSub>batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_4/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�2-
+batch_normalization_4/AssignMovingAvg_1/sub�
+batch_normalization_4/AssignMovingAvg_1/mulMul/batch_normalization_4/AssignMovingAvg_1/sub:z:00batch_normalization_4/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes	
:�2-
+batch_normalization_4/AssignMovingAvg_1/mul�
'batch_normalization_4/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_4_assignmovingavg_1_readvariableop_resource/batch_normalization_4/AssignMovingAvg_1/mul:z:07^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_4/AssignMovingAvg_1�
)batch_normalization_4/Cast/ReadVariableOpReadVariableOp2batch_normalization_4_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_4/Cast/ReadVariableOp�
+batch_normalization_4/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_4_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_4/Cast_1/ReadVariableOp�
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2'
%batch_normalization_4/batchnorm/add/y�
#batch_normalization_4/batchnorm/addAddV20batch_normalization_4/moments/Squeeze_1:output:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_4/batchnorm/add�
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_4/batchnorm/Rsqrt�
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:03batch_normalization_4/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_4/batchnorm/mul�
%batch_normalization_4/batchnorm/mul_1Muldense_3/MatMul:product:0'batch_normalization_4/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_4/batchnorm/mul_1�
%batch_normalization_4/batchnorm/mul_2Mul.batch_normalization_4/moments/Squeeze:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_4/batchnorm/mul_2�
#batch_normalization_4/batchnorm/subSub1batch_normalization_4/Cast/ReadVariableOp:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_4/batchnorm/sub�
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_4/batchnorm/add_1v
Relu_3Relu)batch_normalization_4/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Relu_3�
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_4/MatMul/ReadVariableOp�
dense_4/MatMulMatMulRelu_3:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_4/MatMul�
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_4/BiasAdd/ReadVariableOp�
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_4/BiasAddt
IdentityIdentitydense_4/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identity�
NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp(^batch_normalization/Cast/ReadVariableOp*^batch_normalization/Cast_1/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_1/Cast/ReadVariableOp,^batch_normalization_1/Cast_1/ReadVariableOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_2/Cast/ReadVariableOp,^batch_normalization_2/Cast_1/ReadVariableOp&^batch_normalization_3/AssignMovingAvg5^batch_normalization_3/AssignMovingAvg/ReadVariableOp(^batch_normalization_3/AssignMovingAvg_17^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_3/Cast/ReadVariableOp,^batch_normalization_3/Cast_1/ReadVariableOp&^batch_normalization_4/AssignMovingAvg5^batch_normalization_4/AssignMovingAvg/ReadVariableOp(^batch_normalization_4/AssignMovingAvg_17^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_4/Cast/ReadVariableOp,^batch_normalization_4/Cast_1/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:����������: : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#batch_normalization/AssignMovingAvg#batch_normalization/AssignMovingAvg2h
2batch_normalization/AssignMovingAvg/ReadVariableOp2batch_normalization/AssignMovingAvg/ReadVariableOp2N
%batch_normalization/AssignMovingAvg_1%batch_normalization/AssignMovingAvg_12l
4batch_normalization/AssignMovingAvg_1/ReadVariableOp4batch_normalization/AssignMovingAvg_1/ReadVariableOp2R
'batch_normalization/Cast/ReadVariableOp'batch_normalization/Cast/ReadVariableOp2V
)batch_normalization/Cast_1/ReadVariableOp)batch_normalization/Cast_1/ReadVariableOp2N
%batch_normalization_1/AssignMovingAvg%batch_normalization_1/AssignMovingAvg2l
4batch_normalization_1/AssignMovingAvg/ReadVariableOp4batch_normalization_1/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_1/AssignMovingAvg_1'batch_normalization_1/AssignMovingAvg_12p
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_1/Cast/ReadVariableOp)batch_normalization_1/Cast/ReadVariableOp2Z
+batch_normalization_1/Cast_1/ReadVariableOp+batch_normalization_1/Cast_1/ReadVariableOp2N
%batch_normalization_2/AssignMovingAvg%batch_normalization_2/AssignMovingAvg2l
4batch_normalization_2/AssignMovingAvg/ReadVariableOp4batch_normalization_2/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_2/AssignMovingAvg_1'batch_normalization_2/AssignMovingAvg_12p
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_2/Cast/ReadVariableOp)batch_normalization_2/Cast/ReadVariableOp2Z
+batch_normalization_2/Cast_1/ReadVariableOp+batch_normalization_2/Cast_1/ReadVariableOp2N
%batch_normalization_3/AssignMovingAvg%batch_normalization_3/AssignMovingAvg2l
4batch_normalization_3/AssignMovingAvg/ReadVariableOp4batch_normalization_3/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_3/AssignMovingAvg_1'batch_normalization_3/AssignMovingAvg_12p
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_3/Cast/ReadVariableOp)batch_normalization_3/Cast/ReadVariableOp2Z
+batch_normalization_3/Cast_1/ReadVariableOp+batch_normalization_3/Cast_1/ReadVariableOp2N
%batch_normalization_4/AssignMovingAvg%batch_normalization_4/AssignMovingAvg2l
4batch_normalization_4/AssignMovingAvg/ReadVariableOp4batch_normalization_4/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_4/AssignMovingAvg_1'batch_normalization_4/AssignMovingAvg_12p
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_4/Cast/ReadVariableOp)batch_normalization_4/Cast/ReadVariableOp2Z
+batch_normalization_4/Cast_1/ReadVariableOp+batch_normalization_4/Cast_1/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
<
input_11
serving_default_input_1:0����������=
output_11
StatefulPartitionedCall:0����������tensorflow/serving/predict:��
�
	bn_layers
dense_layers
trainable_variables
regularization_losses
	variables
	keras_api

signatures
+�&call_and_return_all_conditional_losses
�_default_save_signature
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
"15"
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
�
trainable_variables

-layers
regularization_losses
.metrics
	variables
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
#moving_mean
$moving_variance
3trainable_variables
4regularization_losses
5	variables
6	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
7axis
	gamma
beta
%moving_mean
&moving_variance
8trainable_variables
9regularization_losses
:	variables
;	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
<axis
	gamma
beta
'moving_mean
(moving_variance
=trainable_variables
>regularization_losses
?	variables
@	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
Aaxis
	gamma
beta
)moving_mean
*moving_variance
Btrainable_variables
Cregularization_losses
D	variables
E	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
Faxis
	gamma
beta
+moving_mean
,moving_variance
Gtrainable_variables
Hregularization_losses
I	variables
J	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
(
K	keras_api"
_tf_keras_layer
�

kernel
Ltrainable_variables
Mregularization_losses
N	variables
O	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

kernel
Ptrainable_variables
Qregularization_losses
R	variables
S	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

kernel
Ttrainable_variables
Uregularization_losses
V	variables
W	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

 kernel
Xtrainable_variables
Yregularization_losses
Z	variables
[	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

!kernel
"bias
\trainable_variables
]regularization_losses
^	variables
_	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
=:;�2.feed_forward_sub_net/batch_normalization/gamma
<::�2-feed_forward_sub_net/batch_normalization/beta
?:=�20feed_forward_sub_net/batch_normalization_1/gamma
>:<�2/feed_forward_sub_net/batch_normalization_1/beta
?:=�20feed_forward_sub_net/batch_normalization_2/gamma
>:<�2/feed_forward_sub_net/batch_normalization_2/beta
?:=�20feed_forward_sub_net/batch_normalization_3/gamma
>:<�2/feed_forward_sub_net/batch_normalization_3/beta
?:=�20feed_forward_sub_net/batch_normalization_4/gamma
>:<�2/feed_forward_sub_net/batch_normalization_4/beta
5:3
��2!feed_forward_sub_net/dense/kernel
7:5
��2#feed_forward_sub_net/dense_1/kernel
7:5
��2#feed_forward_sub_net/dense_2/kernel
7:5
��2#feed_forward_sub_net/dense_3/kernel
7:5
��2#feed_forward_sub_net/dense_4/kernel
0:.�2!feed_forward_sub_net/dense_4/bias
E:C� (24feed_forward_sub_net/batch_normalization/moving_mean
I:G� (28feed_forward_sub_net/batch_normalization/moving_variance
G:E� (26feed_forward_sub_net/batch_normalization_1/moving_mean
K:I� (2:feed_forward_sub_net/batch_normalization_1/moving_variance
G:E� (26feed_forward_sub_net/batch_normalization_2/moving_mean
K:I� (2:feed_forward_sub_net/batch_normalization_2/moving_variance
G:E� (26feed_forward_sub_net/batch_normalization_3/moving_mean
K:I� (2:feed_forward_sub_net/batch_normalization_3/moving_variance
G:E� (26feed_forward_sub_net/batch_normalization_4/moving_mean
K:I� (2:feed_forward_sub_net/batch_normalization_4/moving_variance
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
�
3trainable_variables

`layers
4regularization_losses
ametrics
5	variables
bnon_trainable_variables
clayer_regularization_losses
dlayer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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
�
8trainable_variables

elayers
9regularization_losses
fmetrics
:	variables
gnon_trainable_variables
hlayer_regularization_losses
ilayer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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
�
=trainable_variables

jlayers
>regularization_losses
kmetrics
?	variables
lnon_trainable_variables
mlayer_regularization_losses
nlayer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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
�
Btrainable_variables

olayers
Cregularization_losses
pmetrics
D	variables
qnon_trainable_variables
rlayer_regularization_losses
slayer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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
�
Gtrainable_variables

tlayers
Hregularization_losses
umetrics
I	variables
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
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
�
Ltrainable_variables

ylayers
Mregularization_losses
zmetrics
N	variables
{non_trainable_variables
|layer_regularization_losses
}layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
�
Ptrainable_variables

~layers
Qregularization_losses
metrics
R	variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
�
Ttrainable_variables
�layers
Uregularization_losses
�metrics
V	variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
'
 0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
 0"
trackable_list_wrapper
�
Xtrainable_variables
�layers
Yregularization_losses
�metrics
Z	variables
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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
�
\trainable_variables
�layers
]regularization_losses
�metrics
^	variables
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
�2�
P__inference_feed_forward_sub_net_layer_call_and_return_conditional_losses_308359
P__inference_feed_forward_sub_net_layer_call_and_return_conditional_losses_308545
P__inference_feed_forward_sub_net_layer_call_and_return_conditional_losses_308651
P__inference_feed_forward_sub_net_layer_call_and_return_conditional_losses_308837�
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
!__inference__wrapped_model_306769input_1"�
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
5__inference_feed_forward_sub_net_layer_call_fn_308894
5__inference_feed_forward_sub_net_layer_call_fn_308951
5__inference_feed_forward_sub_net_layer_call_fn_309008
5__inference_feed_forward_sub_net_layer_call_fn_309065�
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
$__inference_signature_wrapper_308253input_1"�
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
O__inference_batch_normalization_layer_call_and_return_conditional_losses_309085
O__inference_batch_normalization_layer_call_and_return_conditional_losses_309121�
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
4__inference_batch_normalization_layer_call_fn_309134
4__inference_batch_normalization_layer_call_fn_309147�
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
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_309167
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_309203�
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
6__inference_batch_normalization_1_layer_call_fn_309216
6__inference_batch_normalization_1_layer_call_fn_309229�
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
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_309249
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_309285�
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
6__inference_batch_normalization_2_layer_call_fn_309298
6__inference_batch_normalization_2_layer_call_fn_309311�
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
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_309331
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_309367�
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
6__inference_batch_normalization_3_layer_call_fn_309380
6__inference_batch_normalization_3_layer_call_fn_309393�
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
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_309413
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_309449�
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
6__inference_batch_normalization_4_layer_call_fn_309462
6__inference_batch_normalization_4_layer_call_fn_309475�
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
A__inference_dense_layer_call_and_return_conditional_losses_309482�
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
&__inference_dense_layer_call_fn_309489�
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
C__inference_dense_1_layer_call_and_return_conditional_losses_309496�
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
(__inference_dense_1_layer_call_fn_309503�
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
C__inference_dense_2_layer_call_and_return_conditional_losses_309510�
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
(__inference_dense_2_layer_call_fn_309517�
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
C__inference_dense_3_layer_call_and_return_conditional_losses_309524�
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
(__inference_dense_3_layer_call_fn_309531�
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
C__inference_dense_4_layer_call_and_return_conditional_losses_309541�
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
(__inference_dense_4_layer_call_fn_309550�
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
!__inference__wrapped_model_306769�#$%&'()* +,!"1�.
'�$
"�
input_1����������
� "4�1
/
output_1#� 
output_1�����������
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_309167d%&4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_309203d%&4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
6__inference_batch_normalization_1_layer_call_fn_309216W%&4�1
*�'
!�
inputs����������
p 
� "������������
6__inference_batch_normalization_1_layer_call_fn_309229W%&4�1
*�'
!�
inputs����������
p
� "������������
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_309249d'(4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_309285d'(4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
6__inference_batch_normalization_2_layer_call_fn_309298W'(4�1
*�'
!�
inputs����������
p 
� "������������
6__inference_batch_normalization_2_layer_call_fn_309311W'(4�1
*�'
!�
inputs����������
p
� "������������
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_309331d)*4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_309367d)*4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
6__inference_batch_normalization_3_layer_call_fn_309380W)*4�1
*�'
!�
inputs����������
p 
� "������������
6__inference_batch_normalization_3_layer_call_fn_309393W)*4�1
*�'
!�
inputs����������
p
� "������������
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_309413d+,4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_309449d+,4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
6__inference_batch_normalization_4_layer_call_fn_309462W+,4�1
*�'
!�
inputs����������
p 
� "������������
6__inference_batch_normalization_4_layer_call_fn_309475W+,4�1
*�'
!�
inputs����������
p
� "������������
O__inference_batch_normalization_layer_call_and_return_conditional_losses_309085d#$4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
O__inference_batch_normalization_layer_call_and_return_conditional_losses_309121d#$4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
4__inference_batch_normalization_layer_call_fn_309134W#$4�1
*�'
!�
inputs����������
p 
� "������������
4__inference_batch_normalization_layer_call_fn_309147W#$4�1
*�'
!�
inputs����������
p
� "������������
C__inference_dense_1_layer_call_and_return_conditional_losses_309496]0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� |
(__inference_dense_1_layer_call_fn_309503P0�-
&�#
!�
inputs����������
� "������������
C__inference_dense_2_layer_call_and_return_conditional_losses_309510]0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� |
(__inference_dense_2_layer_call_fn_309517P0�-
&�#
!�
inputs����������
� "������������
C__inference_dense_3_layer_call_and_return_conditional_losses_309524] 0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� |
(__inference_dense_3_layer_call_fn_309531P 0�-
&�#
!�
inputs����������
� "������������
C__inference_dense_4_layer_call_and_return_conditional_losses_309541^!"0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� }
(__inference_dense_4_layer_call_fn_309550Q!"0�-
&�#
!�
inputs����������
� "������������
A__inference_dense_layer_call_and_return_conditional_losses_309482]0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� z
&__inference_dense_layer_call_fn_309489P0�-
&�#
!�
inputs����������
� "������������
P__inference_feed_forward_sub_net_layer_call_and_return_conditional_losses_308359u#$%&'()* +,!"/�,
%�"
�
x����������
p 
� "&�#
�
0����������
� �
P__inference_feed_forward_sub_net_layer_call_and_return_conditional_losses_308545u#$%&'()* +,!"/�,
%�"
�
x����������
p
� "&�#
�
0����������
� �
P__inference_feed_forward_sub_net_layer_call_and_return_conditional_losses_308651{#$%&'()* +,!"5�2
+�(
"�
input_1����������
p 
� "&�#
�
0����������
� �
P__inference_feed_forward_sub_net_layer_call_and_return_conditional_losses_308837{#$%&'()* +,!"5�2
+�(
"�
input_1����������
p
� "&�#
�
0����������
� �
5__inference_feed_forward_sub_net_layer_call_fn_308894n#$%&'()* +,!"5�2
+�(
"�
input_1����������
p 
� "������������
5__inference_feed_forward_sub_net_layer_call_fn_308951h#$%&'()* +,!"/�,
%�"
�
x����������
p 
� "������������
5__inference_feed_forward_sub_net_layer_call_fn_309008h#$%&'()* +,!"/�,
%�"
�
x����������
p
� "������������
5__inference_feed_forward_sub_net_layer_call_fn_309065n#$%&'()* +,!"5�2
+�(
"�
input_1����������
p
� "������������
$__inference_signature_wrapper_308253�#$%&'()* +,!"<�9
� 
2�/
-
input_1"�
input_1����������"4�1
/
output_1#� 
output_1����������