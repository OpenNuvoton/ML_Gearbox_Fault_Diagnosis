ɚ
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
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
s
FakeQuantWithMinMaxVars

inputs
min
max
outputs"
num_bitsint"
narrow_rangebool( 
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
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
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
executor_typestring ��
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
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8��
~
Adam/dense_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_9/bias/v
w
'Adam/dense_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_9/kernel/v

)Adam/dense_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_8/bias/v
w
'Adam/dense_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_8/kernel/v

)Adam/dense_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_7/bias/v
w
'Adam/dense_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_7/kernel/v

)Adam/dense_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_6/bias/v
w
'Adam/dense_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_6/kernel/v

)Adam/dense_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_5/bias/v
w
'Adam/dense_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_5/kernel/v

)Adam/dense_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_9/bias/m
w
'Adam/dense_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_9/kernel/m

)Adam/dense_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_8/bias/m
w
'Adam/dense_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_8/kernel/m

)Adam/dense_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_7/bias/m
w
'Adam/dense_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_7/kernel/m

)Adam/dense_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_6/bias/m
w
'Adam/dense_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_6/kernel/m

)Adam/dense_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_5/bias/m
w
'Adam/dense_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_5/kernel/m

)Adam/dense_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/m*
_output_shapes

:*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
p
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_9/bias
i
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes
:*
dtype0
x
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_9/kernel
q
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes

:*
dtype0
p
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_8/bias
i
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes
:*
dtype0
x
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_8/kernel
q
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes

:*
dtype0
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes
:*
dtype0
x
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_7/kernel
q
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes

:*
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes
:*
dtype0
x
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_6/kernel
q
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes

:*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:*
dtype0
x
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

:*
dtype0
�
 quant_dense_9/pre_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" quant_dense_9/pre_activation_max
�
4quant_dense_9/pre_activation_max/Read/ReadVariableOpReadVariableOp quant_dense_9/pre_activation_max*
_output_shapes
: *
dtype0
�
 quant_dense_9/pre_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" quant_dense_9/pre_activation_min
�
4quant_dense_9/pre_activation_min/Read/ReadVariableOpReadVariableOp quant_dense_9/pre_activation_min*
_output_shapes
: *
dtype0
�
quant_dense_9/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namequant_dense_9/kernel_max
}
,quant_dense_9/kernel_max/Read/ReadVariableOpReadVariableOpquant_dense_9/kernel_max*
_output_shapes
: *
dtype0
�
quant_dense_9/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namequant_dense_9/kernel_min
}
,quant_dense_9/kernel_min/Read/ReadVariableOpReadVariableOpquant_dense_9/kernel_min*
_output_shapes
: *
dtype0
�
quant_dense_9/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namequant_dense_9/optimizer_step
�
0quant_dense_9/optimizer_step/Read/ReadVariableOpReadVariableOpquant_dense_9/optimizer_step*
_output_shapes
: *
dtype0
�
quant_activation_7/output_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_activation_7/output_max
�
1quant_activation_7/output_max/Read/ReadVariableOpReadVariableOpquant_activation_7/output_max*
_output_shapes
: *
dtype0
�
quant_activation_7/output_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_activation_7/output_min
�
1quant_activation_7/output_min/Read/ReadVariableOpReadVariableOpquant_activation_7/output_min*
_output_shapes
: *
dtype0
�
!quant_activation_7/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!quant_activation_7/optimizer_step
�
5quant_activation_7/optimizer_step/Read/ReadVariableOpReadVariableOp!quant_activation_7/optimizer_step*
_output_shapes
: *
dtype0
�
quant_dense_8/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namequant_dense_8/optimizer_step
�
0quant_dense_8/optimizer_step/Read/ReadVariableOpReadVariableOpquant_dense_8/optimizer_step*
_output_shapes
: *
dtype0
�
quant_activation_6/output_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_activation_6/output_max
�
1quant_activation_6/output_max/Read/ReadVariableOpReadVariableOpquant_activation_6/output_max*
_output_shapes
: *
dtype0
�
quant_activation_6/output_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_activation_6/output_min
�
1quant_activation_6/output_min/Read/ReadVariableOpReadVariableOpquant_activation_6/output_min*
_output_shapes
: *
dtype0
�
!quant_activation_6/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!quant_activation_6/optimizer_step
�
5quant_activation_6/optimizer_step/Read/ReadVariableOpReadVariableOp!quant_activation_6/optimizer_step*
_output_shapes
: *
dtype0
�
quant_dense_7/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namequant_dense_7/optimizer_step
�
0quant_dense_7/optimizer_step/Read/ReadVariableOpReadVariableOpquant_dense_7/optimizer_step*
_output_shapes
: *
dtype0
�
quant_activation_5/output_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_activation_5/output_max
�
1quant_activation_5/output_max/Read/ReadVariableOpReadVariableOpquant_activation_5/output_max*
_output_shapes
: *
dtype0
�
quant_activation_5/output_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_activation_5/output_min
�
1quant_activation_5/output_min/Read/ReadVariableOpReadVariableOpquant_activation_5/output_min*
_output_shapes
: *
dtype0
�
!quant_activation_5/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!quant_activation_5/optimizer_step
�
5quant_activation_5/optimizer_step/Read/ReadVariableOpReadVariableOp!quant_activation_5/optimizer_step*
_output_shapes
: *
dtype0
�
quant_dense_6/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namequant_dense_6/optimizer_step
�
0quant_dense_6/optimizer_step/Read/ReadVariableOpReadVariableOpquant_dense_6/optimizer_step*
_output_shapes
: *
dtype0
�
quant_activation_4/output_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_activation_4/output_max
�
1quant_activation_4/output_max/Read/ReadVariableOpReadVariableOpquant_activation_4/output_max*
_output_shapes
: *
dtype0
�
quant_activation_4/output_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_activation_4/output_min
�
1quant_activation_4/output_min/Read/ReadVariableOpReadVariableOpquant_activation_4/output_min*
_output_shapes
: *
dtype0
�
!quant_activation_4/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!quant_activation_4/optimizer_step
�
5quant_activation_4/optimizer_step/Read/ReadVariableOpReadVariableOp!quant_activation_4/optimizer_step*
_output_shapes
: *
dtype0
�
quant_dense_5/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namequant_dense_5/optimizer_step
�
0quant_dense_5/optimizer_step/Read/ReadVariableOpReadVariableOpquant_dense_5/optimizer_step*
_output_shapes
: *
dtype0
�
quantize_layer_2/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!quantize_layer_2/optimizer_step
�
3quantize_layer_2/optimizer_step/Read/ReadVariableOpReadVariableOpquantize_layer_2/optimizer_step*
_output_shapes
: *
dtype0
�
%quantize_layer_2/quantize_layer_2_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%quantize_layer_2/quantize_layer_2_max
�
9quantize_layer_2/quantize_layer_2_max/Read/ReadVariableOpReadVariableOp%quantize_layer_2/quantize_layer_2_max*
_output_shapes
: *
dtype0
�
%quantize_layer_2/quantize_layer_2_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%quantize_layer_2/quantize_layer_2_min
�
9quantize_layer_2/quantize_layer_2_min/Read/ReadVariableOpReadVariableOp%quantize_layer_2/quantize_layer_2_min*
_output_shapes
: *
dtype0
�
serving_default_dense_5_inputPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_5_input%quantize_layer_2/quantize_layer_2_min%quantize_layer_2/quantize_layer_2_maxdense_5/kerneldense_5/biasquant_activation_4/output_minquant_activation_4/output_maxdense_6/kerneldense_6/biasquant_activation_5/output_minquant_activation_5/output_maxdense_7/kerneldense_7/biasquant_activation_6/output_minquant_activation_6/output_maxdense_8/kerneldense_8/biasquant_activation_7/output_minquant_activation_7/output_maxdense_9/kernelquant_dense_9/kernel_minquant_dense_9/kernel_maxdense_9/bias quant_dense_9/pre_activation_min quant_dense_9/pre_activation_max*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference_signature_wrapper_174099

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value�B� B�
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
layer_with_weights-6
layer-6
layer_with_weights-7
layer-7
	layer_with_weights-8
	layer-8

layer_with_weights-9

layer-9
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
quantize_layer_2_min
quantize_layer_2_max
quantizer_vars
optimizer_step*
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses
	$layer
%optimizer_step
&_weight_vars
'_quantize_activations
(_output_quantizers*
�
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses
	/layer
0optimizer_step
1_weight_vars
2_quantize_activations
3_output_quantizers
4
output_min
5
output_max
6_output_quantizer_vars*
�
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses
	=layer
>optimizer_step
?_weight_vars
@_quantize_activations
A_output_quantizers*
�
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses
	Hlayer
Ioptimizer_step
J_weight_vars
K_quantize_activations
L_output_quantizers
M
output_min
N
output_max
O_output_quantizer_vars*
�
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses
	Vlayer
Woptimizer_step
X_weight_vars
Y_quantize_activations
Z_output_quantizers*
�
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses
	alayer
boptimizer_step
c_weight_vars
d_quantize_activations
e_output_quantizers
f
output_min
g
output_max
h_output_quantizer_vars*
�
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses
	olayer
poptimizer_step
q_weight_vars
r_quantize_activations
s_output_quantizers*
�
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses
	zlayer
{optimizer_step
|_weight_vars
}_quantize_activations
~_output_quantizers

output_min
�
output_max
�_output_quantizer_vars*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�layer
�optimizer_step
�_weight_vars
�
kernel_min
�
kernel_max
�_quantize_activations
�pre_activation_min
�pre_activation_max
�_output_quantizers*
�
0
1
2
�3
�4
%5
06
47
58
�9
�10
>11
I12
M13
N14
�15
�16
W17
b18
f19
g20
�21
�22
p23
{24
25
�26
�27
�28
�29
�30
�31
�32
�33*
T
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
* 
�
	�iter
�beta_1
�beta_2

�decay
�learning_rate	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�*

�serving_default* 

0
1
2*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�}
VARIABLE_VALUE%quantize_layer_2/quantize_layer_2_minDlayer_with_weights-0/quantize_layer_2_min/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUE%quantize_layer_2/quantize_layer_2_maxDlayer_with_weights-0/quantize_layer_2_max/.ATTRIBUTES/VARIABLE_VALUE*

min_var
max_var*
wq
VARIABLE_VALUEquantize_layer_2/optimizer_step>layer_with_weights-0/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1
%2*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
tn
VARIABLE_VALUEquant_dense_5/optimizer_step>layer_with_weights-1/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

00
41
52*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
ys
VARIABLE_VALUE!quant_activation_4/optimizer_step>layer_with_weights-2/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
qk
VARIABLE_VALUEquant_activation_4/output_min:layer_with_weights-2/output_min/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEquant_activation_4/output_max:layer_with_weights-2/output_max/.ATTRIBUTES/VARIABLE_VALUE*

4min_var
5max_var*

�0
�1
>2*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
tn
VARIABLE_VALUEquant_dense_6/optimizer_step>layer_with_weights-3/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

I0
M1
N2*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
ys
VARIABLE_VALUE!quant_activation_5/optimizer_step>layer_with_weights-4/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
qk
VARIABLE_VALUEquant_activation_5/output_min:layer_with_weights-4/output_min/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEquant_activation_5/output_max:layer_with_weights-4/output_max/.ATTRIBUTES/VARIABLE_VALUE*

Mmin_var
Nmax_var*

�0
�1
W2*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
tn
VARIABLE_VALUEquant_dense_7/optimizer_step>layer_with_weights-5/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

b0
f1
g2*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
ys
VARIABLE_VALUE!quant_activation_6/optimizer_step>layer_with_weights-6/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
qk
VARIABLE_VALUEquant_activation_6/output_min:layer_with_weights-6/output_min/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEquant_activation_6/output_max:layer_with_weights-6/output_max/.ATTRIBUTES/VARIABLE_VALUE*

fmin_var
gmax_var*

�0
�1
p2*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
tn
VARIABLE_VALUEquant_dense_8/optimizer_step>layer_with_weights-7/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

{0
1
�2*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
ys
VARIABLE_VALUE!quant_activation_7/optimizer_step>layer_with_weights-8/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
qk
VARIABLE_VALUEquant_activation_7/output_min:layer_with_weights-8/output_min/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEquant_activation_7/output_max:layer_with_weights-8/output_max/.ATTRIBUTES/VARIABLE_VALUE*

min_var
�max_var*
<
�0
�1
�2
�3
�4
�5
�6*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
tn
VARIABLE_VALUEquant_dense_9/optimizer_step>layer_with_weights-9/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE*

�0*
lf
VARIABLE_VALUEquant_dense_9/kernel_min:layer_with_weights-9/kernel_min/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEquant_dense_9/kernel_max:layer_with_weights-9/kernel_max/.ATTRIBUTES/VARIABLE_VALUE*
* 
|v
VARIABLE_VALUE quant_dense_9/pre_activation_minBlayer_with_weights-9/pre_activation_min/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE quant_dense_9/pre_activation_maxBlayer_with_weights-9/pre_activation_max/.ATTRIBUTES/VARIABLE_VALUE*
* 
NH
VARIABLE_VALUEdense_5/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense_5/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_6/kernel&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_6/bias'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_7/kernel'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_7/bias'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_8/kernel'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_8/bias'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_9/kernel'variables/27/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_9/bias'variables/28/.ATTRIBUTES/VARIABLE_VALUE*
�
0
1
2
%3
04
45
56
>7
I8
M9
N10
W11
b12
f13
g14
p15
{16
17
�18
�19
�20
�21
�22
�23*
J
0
1
2
3
4
5
6
7
	8

9*

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*
* 
* 
* 
* 
* 
* 
* 
* 

%0*

$0*
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 

00
41
52*
	
/0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 

>0*

=0*
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 

I0
M1
N2*
	
H0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 

W0*

V0*
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 

b0
f1
g2*
	
a0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 

p0*

o0*
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 

{0
1
�2*
	
z0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
,
�0
�1
�2
�3
�4*

�0*
* 
* 
* 
* 
* 
* 
* 

�0*

�0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 

�0
�2*
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
 
�min_var
�max_var*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
qk
VARIABLE_VALUEAdam/dense_5/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUEAdam/dense_5/bias/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_6/kernel/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_6/bias/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_7/kernel/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_7/bias/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_8/kernel/mCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_8/bias/mCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_9/kernel/mCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_9/bias/mCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_5/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUEAdam/dense_5/bias/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_6/kernel/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_6/bias/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_7/kernel/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_7/bias/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_8/kernel/vCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_8/bias/vCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_9/kernel/vCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_9/bias/vCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename9quantize_layer_2/quantize_layer_2_min/Read/ReadVariableOp9quantize_layer_2/quantize_layer_2_max/Read/ReadVariableOp3quantize_layer_2/optimizer_step/Read/ReadVariableOp0quant_dense_5/optimizer_step/Read/ReadVariableOp5quant_activation_4/optimizer_step/Read/ReadVariableOp1quant_activation_4/output_min/Read/ReadVariableOp1quant_activation_4/output_max/Read/ReadVariableOp0quant_dense_6/optimizer_step/Read/ReadVariableOp5quant_activation_5/optimizer_step/Read/ReadVariableOp1quant_activation_5/output_min/Read/ReadVariableOp1quant_activation_5/output_max/Read/ReadVariableOp0quant_dense_7/optimizer_step/Read/ReadVariableOp5quant_activation_6/optimizer_step/Read/ReadVariableOp1quant_activation_6/output_min/Read/ReadVariableOp1quant_activation_6/output_max/Read/ReadVariableOp0quant_dense_8/optimizer_step/Read/ReadVariableOp5quant_activation_7/optimizer_step/Read/ReadVariableOp1quant_activation_7/output_min/Read/ReadVariableOp1quant_activation_7/output_max/Read/ReadVariableOp0quant_dense_9/optimizer_step/Read/ReadVariableOp,quant_dense_9/kernel_min/Read/ReadVariableOp,quant_dense_9/kernel_max/Read/ReadVariableOp4quant_dense_9/pre_activation_min/Read/ReadVariableOp4quant_dense_9/pre_activation_max/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOp"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOp"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp)Adam/dense_5/kernel/m/Read/ReadVariableOp'Adam/dense_5/bias/m/Read/ReadVariableOp)Adam/dense_6/kernel/m/Read/ReadVariableOp'Adam/dense_6/bias/m/Read/ReadVariableOp)Adam/dense_7/kernel/m/Read/ReadVariableOp'Adam/dense_7/bias/m/Read/ReadVariableOp)Adam/dense_8/kernel/m/Read/ReadVariableOp'Adam/dense_8/bias/m/Read/ReadVariableOp)Adam/dense_9/kernel/m/Read/ReadVariableOp'Adam/dense_9/bias/m/Read/ReadVariableOp)Adam/dense_5/kernel/v/Read/ReadVariableOp'Adam/dense_5/bias/v/Read/ReadVariableOp)Adam/dense_6/kernel/v/Read/ReadVariableOp'Adam/dense_6/bias/v/Read/ReadVariableOp)Adam/dense_7/kernel/v/Read/ReadVariableOp'Adam/dense_7/bias/v/Read/ReadVariableOp)Adam/dense_8/kernel/v/Read/ReadVariableOp'Adam/dense_8/bias/v/Read/ReadVariableOp)Adam/dense_9/kernel/v/Read/ReadVariableOp'Adam/dense_9/bias/v/Read/ReadVariableOpConst*L
TinE
C2A	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *(
f#R!
__inference__traced_save_175189
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename%quantize_layer_2/quantize_layer_2_min%quantize_layer_2/quantize_layer_2_maxquantize_layer_2/optimizer_stepquant_dense_5/optimizer_step!quant_activation_4/optimizer_stepquant_activation_4/output_minquant_activation_4/output_maxquant_dense_6/optimizer_step!quant_activation_5/optimizer_stepquant_activation_5/output_minquant_activation_5/output_maxquant_dense_7/optimizer_step!quant_activation_6/optimizer_stepquant_activation_6/output_minquant_activation_6/output_maxquant_dense_8/optimizer_step!quant_activation_7/optimizer_stepquant_activation_7/output_minquant_activation_7/output_maxquant_dense_9/optimizer_stepquant_dense_9/kernel_minquant_dense_9/kernel_max quant_dense_9/pre_activation_min quant_dense_9/pre_activation_maxdense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/biasdense_8/kerneldense_8/biasdense_9/kerneldense_9/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/dense_5/kernel/mAdam/dense_5/bias/mAdam/dense_6/kernel/mAdam/dense_6/bias/mAdam/dense_7/kernel/mAdam/dense_7/bias/mAdam/dense_8/kernel/mAdam/dense_8/bias/mAdam/dense_9/kernel/mAdam/dense_9/bias/mAdam/dense_5/kernel/vAdam/dense_5/bias/vAdam/dense_6/kernel/vAdam/dense_6/bias/vAdam/dense_7/kernel/vAdam/dense_7/bias/vAdam/dense_8/kernel/vAdam/dense_8/bias/vAdam/dense_9/kernel/vAdam/dense_9/bias/v*K
TinD
B2@*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__traced_restore_175388��
�
�
L__inference_quantize_layer_2_layer_call_and_return_conditional_losses_173052

inputsK
Aallvaluesquantize_fakequantwithminmaxvars_readvariableop_resource: M
Callvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAallvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCallvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
)AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputs@AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BAllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
IdentityIdentity3AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp9^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 2t
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
N__inference_quant_activation_5_layer_call_and_return_conditional_losses_173116

inputsK
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1F
ReluReluinputs*
T0*'
_output_shapes
:����������
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
I__inference_quant_dense_7_layer_call_and_return_conditional_losses_174727

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
.__inference_quant_dense_5_layer_call_fn_174514

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_quant_dense_5_layer_call_and_return_conditional_losses_173068o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�"
�
L__inference_quantize_layer_2_layer_call_and_return_conditional_losses_173686

inputs;
1allvaluesquantize_minimum_readvariableop_resource: ;
1allvaluesquantize_maximum_readvariableop_resource: 
identity��#AllValuesQuantize/AssignMaxAllValue�#AllValuesQuantize/AssignMinAllValue�8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�(AllValuesQuantize/Maximum/ReadVariableOp�(AllValuesQuantize/Minimum/ReadVariableOph
AllValuesQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       l
AllValuesQuantize/BatchMinMininputs AllValuesQuantize/Const:output:0*
T0*
_output_shapes
: j
AllValuesQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       n
AllValuesQuantize/BatchMaxMaxinputs"AllValuesQuantize/Const_1:output:0*
T0*
_output_shapes
: �
(AllValuesQuantize/Minimum/ReadVariableOpReadVariableOp1allvaluesquantize_minimum_readvariableop_resource*
_output_shapes
: *
dtype0�
AllValuesQuantize/MinimumMinimum0AllValuesQuantize/Minimum/ReadVariableOp:value:0#AllValuesQuantize/BatchMin:output:0*
T0*
_output_shapes
: b
AllValuesQuantize/Minimum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
AllValuesQuantize/Minimum_1MinimumAllValuesQuantize/Minimum:z:0&AllValuesQuantize/Minimum_1/y:output:0*
T0*
_output_shapes
: �
(AllValuesQuantize/Maximum/ReadVariableOpReadVariableOp1allvaluesquantize_maximum_readvariableop_resource*
_output_shapes
: *
dtype0�
AllValuesQuantize/MaximumMaximum0AllValuesQuantize/Maximum/ReadVariableOp:value:0#AllValuesQuantize/BatchMax:output:0*
T0*
_output_shapes
: b
AllValuesQuantize/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
AllValuesQuantize/Maximum_1MaximumAllValuesQuantize/Maximum:z:0&AllValuesQuantize/Maximum_1/y:output:0*
T0*
_output_shapes
: �
#AllValuesQuantize/AssignMinAllValueAssignVariableOp1allvaluesquantize_minimum_readvariableop_resourceAllValuesQuantize/Minimum_1:z:0)^AllValuesQuantize/Minimum/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
#AllValuesQuantize/AssignMaxAllValueAssignVariableOp1allvaluesquantize_maximum_readvariableop_resourceAllValuesQuantize/Maximum_1:z:0)^AllValuesQuantize/Maximum/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp1allvaluesquantize_minimum_readvariableop_resource$^AllValuesQuantize/AssignMinAllValue*
_output_shapes
: *
dtype0�
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp1allvaluesquantize_maximum_readvariableop_resource$^AllValuesQuantize/AssignMaxAllValue*
_output_shapes
: *
dtype0�
)AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputs@AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BAllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
IdentityIdentity3AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp$^AllValuesQuantize/AssignMaxAllValue$^AllValuesQuantize/AssignMinAllValue9^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1)^AllValuesQuantize/Maximum/ReadVariableOp)^AllValuesQuantize/Minimum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 2J
#AllValuesQuantize/AssignMaxAllValue#AllValuesQuantize/AssignMaxAllValue2J
#AllValuesQuantize/AssignMinAllValue#AllValuesQuantize/AssignMinAllValue2t
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12T
(AllValuesQuantize/Maximum/ReadVariableOp(AllValuesQuantize/Maximum/ReadVariableOp2T
(AllValuesQuantize/Minimum/ReadVariableOp(AllValuesQuantize/Minimum/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�;
�

H__inference_sequential_1_layer_call_and_return_conditional_losses_173810

inputs!
quantize_layer_2_173751: !
quantize_layer_2_173753: &
quant_dense_5_173756:"
quant_dense_5_173758:#
quant_activation_4_173761: #
quant_activation_4_173763: &
quant_dense_6_173766:"
quant_dense_6_173768:#
quant_activation_5_173771: #
quant_activation_5_173773: &
quant_dense_7_173776:"
quant_dense_7_173778:#
quant_activation_6_173781: #
quant_activation_6_173783: &
quant_dense_8_173786:"
quant_dense_8_173788:#
quant_activation_7_173791: #
quant_activation_7_173793: &
quant_dense_9_173796:
quant_dense_9_173798: 
quant_dense_9_173800: "
quant_dense_9_173802:
quant_dense_9_173804: 
quant_dense_9_173806: 
identity��*quant_activation_4/StatefulPartitionedCall�*quant_activation_5/StatefulPartitionedCall�*quant_activation_6/StatefulPartitionedCall�*quant_activation_7/StatefulPartitionedCall�%quant_dense_5/StatefulPartitionedCall�%quant_dense_6/StatefulPartitionedCall�%quant_dense_7/StatefulPartitionedCall�%quant_dense_8/StatefulPartitionedCall�%quant_dense_9/StatefulPartitionedCall�(quantize_layer_2/StatefulPartitionedCall�
(quantize_layer_2/StatefulPartitionedCallStatefulPartitionedCallinputsquantize_layer_2_173751quantize_layer_2_173753*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_quantize_layer_2_layer_call_and_return_conditional_losses_173686�
%quant_dense_5/StatefulPartitionedCallStatefulPartitionedCall1quantize_layer_2/StatefulPartitionedCall:output:0quant_dense_5_173756quant_dense_5_173758*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_quant_dense_5_layer_call_and_return_conditional_losses_173646�
*quant_activation_4/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_5/StatefulPartitionedCall:output:0quant_activation_4_173761quant_activation_4_173763*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_quant_activation_4_layer_call_and_return_conditional_losses_173617�
%quant_dense_6/StatefulPartitionedCallStatefulPartitionedCall3quant_activation_4/StatefulPartitionedCall:output:0quant_dense_6_173766quant_dense_6_173768*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_quant_dense_6_layer_call_and_return_conditional_losses_173572�
*quant_activation_5/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_6/StatefulPartitionedCall:output:0quant_activation_5_173771quant_activation_5_173773*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_quant_activation_5_layer_call_and_return_conditional_losses_173543�
%quant_dense_7/StatefulPartitionedCallStatefulPartitionedCall3quant_activation_5/StatefulPartitionedCall:output:0quant_dense_7_173776quant_dense_7_173778*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_quant_dense_7_layer_call_and_return_conditional_losses_173498�
*quant_activation_6/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_7/StatefulPartitionedCall:output:0quant_activation_6_173781quant_activation_6_173783*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_quant_activation_6_layer_call_and_return_conditional_losses_173469�
%quant_dense_8/StatefulPartitionedCallStatefulPartitionedCall3quant_activation_6/StatefulPartitionedCall:output:0quant_dense_8_173786quant_dense_8_173788*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_quant_dense_8_layer_call_and_return_conditional_losses_173424�
*quant_activation_7/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_8/StatefulPartitionedCall:output:0quant_activation_7_173791quant_activation_7_173793*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_quant_activation_7_layer_call_and_return_conditional_losses_173395�
%quant_dense_9/StatefulPartitionedCallStatefulPartitionedCall3quant_activation_7/StatefulPartitionedCall:output:0quant_dense_9_173796quant_dense_9_173798quant_dense_9_173800quant_dense_9_173802quant_dense_9_173804quant_dense_9_173806*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_quant_dense_9_layer_call_and_return_conditional_losses_173342}
IdentityIdentity.quant_dense_9/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp+^quant_activation_4/StatefulPartitionedCall+^quant_activation_5/StatefulPartitionedCall+^quant_activation_6/StatefulPartitionedCall+^quant_activation_7/StatefulPartitionedCall&^quant_dense_5/StatefulPartitionedCall&^quant_dense_6/StatefulPartitionedCall&^quant_dense_7/StatefulPartitionedCall&^quant_dense_8/StatefulPartitionedCall&^quant_dense_9/StatefulPartitionedCall)^quantize_layer_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������: : : : : : : : : : : : : : : : : : : : : : : : 2X
*quant_activation_4/StatefulPartitionedCall*quant_activation_4/StatefulPartitionedCall2X
*quant_activation_5/StatefulPartitionedCall*quant_activation_5/StatefulPartitionedCall2X
*quant_activation_6/StatefulPartitionedCall*quant_activation_6/StatefulPartitionedCall2X
*quant_activation_7/StatefulPartitionedCall*quant_activation_7/StatefulPartitionedCall2N
%quant_dense_5/StatefulPartitionedCall%quant_dense_5/StatefulPartitionedCall2N
%quant_dense_6/StatefulPartitionedCall%quant_dense_6/StatefulPartitionedCall2N
%quant_dense_7/StatefulPartitionedCall%quant_dense_7/StatefulPartitionedCall2N
%quant_dense_8/StatefulPartitionedCall%quant_dense_8/StatefulPartitionedCall2N
%quant_dense_9/StatefulPartitionedCall%quant_dense_9/StatefulPartitionedCall2T
(quantize_layer_2/StatefulPartitionedCall(quantize_layer_2/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
1__inference_quantize_layer_2_layer_call_fn_174475

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_quantize_layer_2_layer_call_and_return_conditional_losses_173686o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
N__inference_quant_activation_4_layer_call_and_return_conditional_losses_174571

inputsK
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1F
ReluReluinputs*
T0*'
_output_shapes
:����������
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
-__inference_sequential_1_layer_call_fn_173273
dense_5_input
unknown: 
	unknown_0: 
	unknown_1:
	unknown_2:
	unknown_3: 
	unknown_4: 
	unknown_5:
	unknown_6:
	unknown_7: 
	unknown_8: 
	unknown_9:

unknown_10:

unknown_11: 

unknown_12: 

unknown_13:

unknown_14:

unknown_15: 

unknown_16: 

unknown_17:

unknown_18: 

unknown_19: 

unknown_20:

unknown_21: 

unknown_22: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_173222o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:���������
'
_user_specified_namedense_5_input
�)
�
N__inference_quant_activation_7_layer_call_and_return_conditional_losses_174873

inputs@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1F
ReluReluinputs*
T0*'
_output_shapes
:���������h
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: j
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0�
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 2h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�;
�

H__inference_sequential_1_layer_call_and_return_conditional_losses_173976
dense_5_input!
quantize_layer_2_173917: !
quantize_layer_2_173919: &
quant_dense_5_173922:"
quant_dense_5_173924:#
quant_activation_4_173927: #
quant_activation_4_173929: &
quant_dense_6_173932:"
quant_dense_6_173934:#
quant_activation_5_173937: #
quant_activation_5_173939: &
quant_dense_7_173942:"
quant_dense_7_173944:#
quant_activation_6_173947: #
quant_activation_6_173949: &
quant_dense_8_173952:"
quant_dense_8_173954:#
quant_activation_7_173957: #
quant_activation_7_173959: &
quant_dense_9_173962:
quant_dense_9_173964: 
quant_dense_9_173966: "
quant_dense_9_173968:
quant_dense_9_173970: 
quant_dense_9_173972: 
identity��*quant_activation_4/StatefulPartitionedCall�*quant_activation_5/StatefulPartitionedCall�*quant_activation_6/StatefulPartitionedCall�*quant_activation_7/StatefulPartitionedCall�%quant_dense_5/StatefulPartitionedCall�%quant_dense_6/StatefulPartitionedCall�%quant_dense_7/StatefulPartitionedCall�%quant_dense_8/StatefulPartitionedCall�%quant_dense_9/StatefulPartitionedCall�(quantize_layer_2/StatefulPartitionedCall�
(quantize_layer_2/StatefulPartitionedCallStatefulPartitionedCalldense_5_inputquantize_layer_2_173917quantize_layer_2_173919*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_quantize_layer_2_layer_call_and_return_conditional_losses_173052�
%quant_dense_5/StatefulPartitionedCallStatefulPartitionedCall1quantize_layer_2/StatefulPartitionedCall:output:0quant_dense_5_173922quant_dense_5_173924*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_quant_dense_5_layer_call_and_return_conditional_losses_173068�
*quant_activation_4/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_5/StatefulPartitionedCall:output:0quant_activation_4_173927quant_activation_4_173929*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_quant_activation_4_layer_call_and_return_conditional_losses_173084�
%quant_dense_6/StatefulPartitionedCallStatefulPartitionedCall3quant_activation_4/StatefulPartitionedCall:output:0quant_dense_6_173932quant_dense_6_173934*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_quant_dense_6_layer_call_and_return_conditional_losses_173100�
*quant_activation_5/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_6/StatefulPartitionedCall:output:0quant_activation_5_173937quant_activation_5_173939*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_quant_activation_5_layer_call_and_return_conditional_losses_173116�
%quant_dense_7/StatefulPartitionedCallStatefulPartitionedCall3quant_activation_5/StatefulPartitionedCall:output:0quant_dense_7_173942quant_dense_7_173944*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_quant_dense_7_layer_call_and_return_conditional_losses_173132�
*quant_activation_6/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_7/StatefulPartitionedCall:output:0quant_activation_6_173947quant_activation_6_173949*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_quant_activation_6_layer_call_and_return_conditional_losses_173148�
%quant_dense_8/StatefulPartitionedCallStatefulPartitionedCall3quant_activation_6/StatefulPartitionedCall:output:0quant_dense_8_173952quant_dense_8_173954*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_quant_dense_8_layer_call_and_return_conditional_losses_173164�
*quant_activation_7/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_8/StatefulPartitionedCall:output:0quant_activation_7_173957quant_activation_7_173959*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_quant_activation_7_layer_call_and_return_conditional_losses_173180�
%quant_dense_9/StatefulPartitionedCallStatefulPartitionedCall3quant_activation_7/StatefulPartitionedCall:output:0quant_dense_9_173962quant_dense_9_173964quant_dense_9_173966quant_dense_9_173968quant_dense_9_173970quant_dense_9_173972*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_quant_dense_9_layer_call_and_return_conditional_losses_173207}
IdentityIdentity.quant_dense_9/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp+^quant_activation_4/StatefulPartitionedCall+^quant_activation_5/StatefulPartitionedCall+^quant_activation_6/StatefulPartitionedCall+^quant_activation_7/StatefulPartitionedCall&^quant_dense_5/StatefulPartitionedCall&^quant_dense_6/StatefulPartitionedCall&^quant_dense_7/StatefulPartitionedCall&^quant_dense_8/StatefulPartitionedCall&^quant_dense_9/StatefulPartitionedCall)^quantize_layer_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������: : : : : : : : : : : : : : : : : : : : : : : : 2X
*quant_activation_4/StatefulPartitionedCall*quant_activation_4/StatefulPartitionedCall2X
*quant_activation_5/StatefulPartitionedCall*quant_activation_5/StatefulPartitionedCall2X
*quant_activation_6/StatefulPartitionedCall*quant_activation_6/StatefulPartitionedCall2X
*quant_activation_7/StatefulPartitionedCall*quant_activation_7/StatefulPartitionedCall2N
%quant_dense_5/StatefulPartitionedCall%quant_dense_5/StatefulPartitionedCall2N
%quant_dense_6/StatefulPartitionedCall%quant_dense_6/StatefulPartitionedCall2N
%quant_dense_7/StatefulPartitionedCall%quant_dense_7/StatefulPartitionedCall2N
%quant_dense_8/StatefulPartitionedCall%quant_dense_8/StatefulPartitionedCall2N
%quant_dense_9/StatefulPartitionedCall%quant_dense_9/StatefulPartitionedCall2T
(quantize_layer_2/StatefulPartitionedCall(quantize_layer_2/StatefulPartitionedCall:V R
'
_output_shapes
:���������
'
_user_specified_namedense_5_input
�	
�
I__inference_quant_dense_6_layer_call_and_return_conditional_losses_174635

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
3__inference_quant_activation_4_layer_call_fn_174561

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_quant_activation_4_layer_call_and_return_conditional_losses_173617o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
I__inference_quant_dense_5_layer_call_and_return_conditional_losses_173646

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_174099
dense_5_input
unknown: 
	unknown_0: 
	unknown_1:
	unknown_2:
	unknown_3: 
	unknown_4: 
	unknown_5:
	unknown_6:
	unknown_7: 
	unknown_8: 
	unknown_9:

unknown_10:

unknown_11: 

unknown_12: 

unknown_13:

unknown_14:

unknown_15: 

unknown_16: 

unknown_17:

unknown_18: 

unknown_19: 

unknown_20:

unknown_21: 

unknown_22: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__wrapped_model_173036o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:���������
'
_user_specified_namedense_5_input
�
�
3__inference_quant_activation_4_layer_call_fn_174552

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_quant_activation_4_layer_call_and_return_conditional_losses_173084o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
N__inference_quant_activation_6_layer_call_and_return_conditional_losses_174755

inputsK
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1F
ReluReluinputs*
T0*'
_output_shapes
:����������
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
.__inference_quant_dense_6_layer_call_fn_174615

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_quant_dense_6_layer_call_and_return_conditional_losses_173572o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
N__inference_quant_activation_7_layer_call_and_return_conditional_losses_174847

inputsK
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1F
ReluReluinputs*
T0*'
_output_shapes
:����������
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
� 
�
I__inference_quant_dense_9_layer_call_and_return_conditional_losses_173207

inputsP
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: -
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��BiasAdd/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:*
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsBiasAdd:output:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������y
SoftmaxSoftmax3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
� 
�
I__inference_quant_dense_9_layer_call_and_return_conditional_losses_174928

inputsP
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: -
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��BiasAdd/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:*
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsBiasAdd:output:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������y
SoftmaxSoftmax3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�)
�
N__inference_quant_activation_5_layer_call_and_return_conditional_losses_174689

inputs@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1F
ReluReluinputs*
T0*'
_output_shapes
:���������h
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: j
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0�
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 2h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�)
�
N__inference_quant_activation_4_layer_call_and_return_conditional_losses_173617

inputs@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1F
ReluReluinputs*
T0*'
_output_shapes
:���������h
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: j
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0�
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 2h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
.__inference_quant_dense_8_layer_call_fn_174790

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_quant_dense_8_layer_call_and_return_conditional_losses_173164o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
I__inference_quant_dense_5_layer_call_and_return_conditional_losses_174533

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
-__inference_sequential_1_layer_call_fn_173914
dense_5_input
unknown: 
	unknown_0: 
	unknown_1:
	unknown_2:
	unknown_3: 
	unknown_4: 
	unknown_5:
	unknown_6:
	unknown_7: 
	unknown_8: 
	unknown_9:

unknown_10:

unknown_11: 

unknown_12: 

unknown_13:

unknown_14:

unknown_15: 

unknown_16: 

unknown_17:

unknown_18: 

unknown_19: 

unknown_20:

unknown_21: 

unknown_22: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_173810o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:���������
'
_user_specified_namedense_5_input
�
�
.__inference_quant_dense_5_layer_call_fn_174523

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_quant_dense_5_layer_call_and_return_conditional_losses_173646o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
N__inference_quant_activation_7_layer_call_and_return_conditional_losses_173180

inputsK
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1F
ReluReluinputs*
T0*'
_output_shapes
:����������
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�)
�
N__inference_quant_activation_5_layer_call_and_return_conditional_losses_173543

inputs@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1F
ReluReluinputs*
T0*'
_output_shapes
:���������h
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: j
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0�
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 2h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
I__inference_quant_dense_5_layer_call_and_return_conditional_losses_173068

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
L__inference_quantize_layer_2_layer_call_and_return_conditional_losses_174484

inputsK
Aallvaluesquantize_fakequantwithminmaxvars_readvariableop_resource: M
Callvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAallvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCallvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
)AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputs@AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BAllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
IdentityIdentity3AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp9^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 2t
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
.__inference_quant_dense_9_layer_call_fn_174907

inputs
unknown:
	unknown_0: 
	unknown_1: 
	unknown_2:
	unknown_3: 
	unknown_4: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_quant_dense_9_layer_call_and_return_conditional_losses_173342o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
I__inference_quant_dense_5_layer_call_and_return_conditional_losses_174543

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
ɣ
� 
!__inference__wrapped_model_173036
dense_5_inputi
_sequential_1_quantize_layer_2_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource: k
asequential_1_quantize_layer_2_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource: K
9sequential_1_quant_dense_5_matmul_readvariableop_resource:H
:sequential_1_quant_dense_5_biasadd_readvariableop_resource:k
asequential_1_quant_activation_4_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: m
csequential_1_quant_activation_4_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: K
9sequential_1_quant_dense_6_matmul_readvariableop_resource:H
:sequential_1_quant_dense_6_biasadd_readvariableop_resource:k
asequential_1_quant_activation_5_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: m
csequential_1_quant_activation_5_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: K
9sequential_1_quant_dense_7_matmul_readvariableop_resource:H
:sequential_1_quant_dense_7_biasadd_readvariableop_resource:k
asequential_1_quant_activation_6_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: m
csequential_1_quant_activation_6_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: K
9sequential_1_quant_dense_8_matmul_readvariableop_resource:H
:sequential_1_quant_dense_8_biasadd_readvariableop_resource:k
asequential_1_quant_activation_7_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: m
csequential_1_quant_activation_7_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: k
Ysequential_1_quant_dense_9_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:e
[sequential_1_quant_dense_9_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: e
[sequential_1_quant_dense_9_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: H
:sequential_1_quant_dense_9_biasadd_readvariableop_resource:f
\sequential_1_quant_dense_9_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: h
^sequential_1_quant_dense_9_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��Xsequential_1/quant_activation_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Zsequential_1/quant_activation_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�Xsequential_1/quant_activation_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Zsequential_1/quant_activation_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�Xsequential_1/quant_activation_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Zsequential_1/quant_activation_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�Xsequential_1/quant_activation_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Zsequential_1/quant_activation_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�1sequential_1/quant_dense_5/BiasAdd/ReadVariableOp�0sequential_1/quant_dense_5/MatMul/ReadVariableOp�1sequential_1/quant_dense_6/BiasAdd/ReadVariableOp�0sequential_1/quant_dense_6/MatMul/ReadVariableOp�1sequential_1/quant_dense_7/BiasAdd/ReadVariableOp�0sequential_1/quant_dense_7/MatMul/ReadVariableOp�1sequential_1/quant_dense_8/BiasAdd/ReadVariableOp�0sequential_1/quant_dense_8/MatMul/ReadVariableOp�1sequential_1/quant_dense_9/BiasAdd/ReadVariableOp�Psequential_1/quant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Rsequential_1/quant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Rsequential_1/quant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Ssequential_1/quant_dense_9/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Usequential_1/quant_dense_9/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�Vsequential_1/quantize_layer_2/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Xsequential_1/quantize_layer_2/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
Vsequential_1/quantize_layer_2/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp_sequential_1_quantize_layer_2_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Xsequential_1/quantize_layer_2/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpasequential_1_quantize_layer_2_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Gsequential_1/quantize_layer_2/AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsdense_5_input^sequential_1/quantize_layer_2/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0`sequential_1/quantize_layer_2/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
0sequential_1/quant_dense_5/MatMul/ReadVariableOpReadVariableOp9sequential_1_quant_dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
!sequential_1/quant_dense_5/MatMulMatMulQsequential_1/quantize_layer_2/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:08sequential_1/quant_dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
1sequential_1/quant_dense_5/BiasAdd/ReadVariableOpReadVariableOp:sequential_1_quant_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
"sequential_1/quant_dense_5/BiasAddBiasAdd+sequential_1/quant_dense_5/MatMul:product:09sequential_1/quant_dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
$sequential_1/quant_activation_4/ReluRelu+sequential_1/quant_dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:����������
Xsequential_1/quant_activation_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpasequential_1_quant_activation_4_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Zsequential_1/quant_activation_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpcsequential_1_quant_activation_4_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Isequential_1/quant_activation_4/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars2sequential_1/quant_activation_4/Relu:activations:0`sequential_1/quant_activation_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0bsequential_1/quant_activation_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
0sequential_1/quant_dense_6/MatMul/ReadVariableOpReadVariableOp9sequential_1_quant_dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
!sequential_1/quant_dense_6/MatMulMatMulSsequential_1/quant_activation_4/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:08sequential_1/quant_dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
1sequential_1/quant_dense_6/BiasAdd/ReadVariableOpReadVariableOp:sequential_1_quant_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
"sequential_1/quant_dense_6/BiasAddBiasAdd+sequential_1/quant_dense_6/MatMul:product:09sequential_1/quant_dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
$sequential_1/quant_activation_5/ReluRelu+sequential_1/quant_dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:����������
Xsequential_1/quant_activation_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpasequential_1_quant_activation_5_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Zsequential_1/quant_activation_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpcsequential_1_quant_activation_5_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Isequential_1/quant_activation_5/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars2sequential_1/quant_activation_5/Relu:activations:0`sequential_1/quant_activation_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0bsequential_1/quant_activation_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
0sequential_1/quant_dense_7/MatMul/ReadVariableOpReadVariableOp9sequential_1_quant_dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
!sequential_1/quant_dense_7/MatMulMatMulSsequential_1/quant_activation_5/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:08sequential_1/quant_dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
1sequential_1/quant_dense_7/BiasAdd/ReadVariableOpReadVariableOp:sequential_1_quant_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
"sequential_1/quant_dense_7/BiasAddBiasAdd+sequential_1/quant_dense_7/MatMul:product:09sequential_1/quant_dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
$sequential_1/quant_activation_6/ReluRelu+sequential_1/quant_dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:����������
Xsequential_1/quant_activation_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpasequential_1_quant_activation_6_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Zsequential_1/quant_activation_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpcsequential_1_quant_activation_6_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Isequential_1/quant_activation_6/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars2sequential_1/quant_activation_6/Relu:activations:0`sequential_1/quant_activation_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0bsequential_1/quant_activation_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
0sequential_1/quant_dense_8/MatMul/ReadVariableOpReadVariableOp9sequential_1_quant_dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
!sequential_1/quant_dense_8/MatMulMatMulSsequential_1/quant_activation_6/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:08sequential_1/quant_dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
1sequential_1/quant_dense_8/BiasAdd/ReadVariableOpReadVariableOp:sequential_1_quant_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
"sequential_1/quant_dense_8/BiasAddBiasAdd+sequential_1/quant_dense_8/MatMul:product:09sequential_1/quant_dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
$sequential_1/quant_activation_7/ReluRelu+sequential_1/quant_dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:����������
Xsequential_1/quant_activation_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpasequential_1_quant_activation_7_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Zsequential_1/quant_activation_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpcsequential_1_quant_activation_7_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Isequential_1/quant_activation_7/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars2sequential_1/quant_activation_7/Relu:activations:0`sequential_1/quant_activation_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0bsequential_1/quant_activation_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
Psequential_1/quant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpYsequential_1_quant_dense_9_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:*
dtype0�
Rsequential_1/quant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp[sequential_1_quant_dense_9_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Rsequential_1/quant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp[sequential_1_quant_dense_9_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
Asequential_1/quant_dense_9/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsXsequential_1/quant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Zsequential_1/quant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Zsequential_1/quant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:*
narrow_range(�
!sequential_1/quant_dense_9/MatMulMatMulSsequential_1/quant_activation_7/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Ksequential_1/quant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:����������
1sequential_1/quant_dense_9/BiasAdd/ReadVariableOpReadVariableOp:sequential_1_quant_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
"sequential_1/quant_dense_9/BiasAddBiasAdd+sequential_1/quant_dense_9/MatMul:product:09sequential_1/quant_dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
Ssequential_1/quant_dense_9/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp\sequential_1_quant_dense_9_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Usequential_1/quant_dense_9/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp^sequential_1_quant_dense_9_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Dsequential_1/quant_dense_9/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars+sequential_1/quant_dense_9/BiasAdd:output:0[sequential_1/quant_dense_9/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0]sequential_1/quant_dense_9/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
"sequential_1/quant_dense_9/SoftmaxSoftmaxNsequential_1/quant_dense_9/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������{
IdentityIdentity,sequential_1/quant_dense_9/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOpY^sequential_1/quant_activation_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp[^sequential_1/quant_activation_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Y^sequential_1/quant_activation_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp[^sequential_1/quant_activation_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Y^sequential_1/quant_activation_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp[^sequential_1/quant_activation_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Y^sequential_1/quant_activation_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp[^sequential_1/quant_activation_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12^sequential_1/quant_dense_5/BiasAdd/ReadVariableOp1^sequential_1/quant_dense_5/MatMul/ReadVariableOp2^sequential_1/quant_dense_6/BiasAdd/ReadVariableOp1^sequential_1/quant_dense_6/MatMul/ReadVariableOp2^sequential_1/quant_dense_7/BiasAdd/ReadVariableOp1^sequential_1/quant_dense_7/MatMul/ReadVariableOp2^sequential_1/quant_dense_8/BiasAdd/ReadVariableOp1^sequential_1/quant_dense_8/MatMul/ReadVariableOp2^sequential_1/quant_dense_9/BiasAdd/ReadVariableOpQ^sequential_1/quant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpS^sequential_1/quant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1S^sequential_1/quant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2T^sequential_1/quant_dense_9/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpV^sequential_1/quant_dense_9/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1W^sequential_1/quantize_layer_2/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpY^sequential_1/quantize_layer_2/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������: : : : : : : : : : : : : : : : : : : : : : : : 2�
Xsequential_1/quant_activation_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpXsequential_1/quant_activation_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Zsequential_1/quant_activation_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Zsequential_1/quant_activation_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Xsequential_1/quant_activation_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpXsequential_1/quant_activation_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Zsequential_1/quant_activation_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Zsequential_1/quant_activation_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Xsequential_1/quant_activation_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpXsequential_1/quant_activation_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Zsequential_1/quant_activation_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Zsequential_1/quant_activation_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Xsequential_1/quant_activation_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpXsequential_1/quant_activation_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Zsequential_1/quant_activation_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Zsequential_1/quant_activation_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12f
1sequential_1/quant_dense_5/BiasAdd/ReadVariableOp1sequential_1/quant_dense_5/BiasAdd/ReadVariableOp2d
0sequential_1/quant_dense_5/MatMul/ReadVariableOp0sequential_1/quant_dense_5/MatMul/ReadVariableOp2f
1sequential_1/quant_dense_6/BiasAdd/ReadVariableOp1sequential_1/quant_dense_6/BiasAdd/ReadVariableOp2d
0sequential_1/quant_dense_6/MatMul/ReadVariableOp0sequential_1/quant_dense_6/MatMul/ReadVariableOp2f
1sequential_1/quant_dense_7/BiasAdd/ReadVariableOp1sequential_1/quant_dense_7/BiasAdd/ReadVariableOp2d
0sequential_1/quant_dense_7/MatMul/ReadVariableOp0sequential_1/quant_dense_7/MatMul/ReadVariableOp2f
1sequential_1/quant_dense_8/BiasAdd/ReadVariableOp1sequential_1/quant_dense_8/BiasAdd/ReadVariableOp2d
0sequential_1/quant_dense_8/MatMul/ReadVariableOp0sequential_1/quant_dense_8/MatMul/ReadVariableOp2f
1sequential_1/quant_dense_9/BiasAdd/ReadVariableOp1sequential_1/quant_dense_9/BiasAdd/ReadVariableOp2�
Psequential_1/quant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpPsequential_1/quant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Rsequential_1/quant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Rsequential_1/quant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Rsequential_1/quant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Rsequential_1/quant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Ssequential_1/quant_dense_9/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpSsequential_1/quant_dense_9/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Usequential_1/quant_dense_9/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Usequential_1/quant_dense_9/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Vsequential_1/quantize_layer_2/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpVsequential_1/quantize_layer_2/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Xsequential_1/quantize_layer_2/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Xsequential_1/quantize_layer_2/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:V R
'
_output_shapes
:���������
'
_user_specified_namedense_5_input
�N
�
I__inference_quant_dense_9_layer_call_and_return_conditional_losses_174977

inputsA
/lastvaluequant_batchmin_readvariableop_resource:/
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�LastValueQuant/AssignMaxLast�LastValueQuant/AssignMinLast�&LastValueQuant/BatchMax/ReadVariableOp�&LastValueQuant/BatchMin/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1e
LastValueQuant/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*
_output_shapes

:*
dtype0�
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/Const:output:0*
T0*
_output_shapes
: g
LastValueQuant/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*
_output_shapes

:*
dtype0�
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:0LastValueQuant/Const_1:output:0*
T0*
_output_shapes
: ]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: �
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
: Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��{
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: |
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
: �
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(�
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*
_output_shapes

:*
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0�
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       v
MovingAvgQuantize/BatchMinMinBiasAdd:output:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: j
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       x
MovingAvgQuantize/BatchMaxMaxBiasAdd:output:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0�
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsBiasAdd:output:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������y
SoftmaxSoftmax3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
I__inference_quant_dense_8_layer_call_and_return_conditional_losses_174819

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
I__inference_quant_dense_6_layer_call_and_return_conditional_losses_173100

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
N__inference_quant_activation_4_layer_call_and_return_conditional_losses_173084

inputsK
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1F
ReluReluinputs*
T0*'
_output_shapes
:����������
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
.__inference_quant_dense_8_layer_call_fn_174799

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_quant_dense_8_layer_call_and_return_conditional_losses_173424o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
3__inference_quant_activation_7_layer_call_fn_174828

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_quant_activation_7_layer_call_and_return_conditional_losses_173180o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�)
�
N__inference_quant_activation_4_layer_call_and_return_conditional_losses_174597

inputs@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1F
ReluReluinputs*
T0*'
_output_shapes
:���������h
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: j
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0�
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 2h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�(
H__inference_sequential_1_layer_call_and_return_conditional_losses_174457

inputsL
Bquantize_layer_2_allvaluesquantize_minimum_readvariableop_resource: L
Bquantize_layer_2_allvaluesquantize_maximum_readvariableop_resource: >
,quant_dense_5_matmul_readvariableop_resource:;
-quant_dense_5_biasadd_readvariableop_resource:S
Iquant_activation_4_movingavgquantize_assignminema_readvariableop_resource: S
Iquant_activation_4_movingavgquantize_assignmaxema_readvariableop_resource: >
,quant_dense_6_matmul_readvariableop_resource:;
-quant_dense_6_biasadd_readvariableop_resource:S
Iquant_activation_5_movingavgquantize_assignminema_readvariableop_resource: S
Iquant_activation_5_movingavgquantize_assignmaxema_readvariableop_resource: >
,quant_dense_7_matmul_readvariableop_resource:;
-quant_dense_7_biasadd_readvariableop_resource:S
Iquant_activation_6_movingavgquantize_assignminema_readvariableop_resource: S
Iquant_activation_6_movingavgquantize_assignmaxema_readvariableop_resource: >
,quant_dense_8_matmul_readvariableop_resource:;
-quant_dense_8_biasadd_readvariableop_resource:S
Iquant_activation_7_movingavgquantize_assignminema_readvariableop_resource: S
Iquant_activation_7_movingavgquantize_assignmaxema_readvariableop_resource: O
=quant_dense_9_lastvaluequant_batchmin_readvariableop_resource:=
3quant_dense_9_lastvaluequant_assignminlast_resource: =
3quant_dense_9_lastvaluequant_assignmaxlast_resource: ;
-quant_dense_9_biasadd_readvariableop_resource:N
Dquant_dense_9_movingavgquantize_assignminema_readvariableop_resource: N
Dquant_dense_9_movingavgquantize_assignmaxema_readvariableop_resource: 
identity��Equant_activation_4/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�@quant_activation_4/MovingAvgQuantize/AssignMaxEma/ReadVariableOp�Equant_activation_4/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�@quant_activation_4/MovingAvgQuantize/AssignMinEma/ReadVariableOp�Kquant_activation_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Mquant_activation_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�Equant_activation_5/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�@quant_activation_5/MovingAvgQuantize/AssignMaxEma/ReadVariableOp�Equant_activation_5/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�@quant_activation_5/MovingAvgQuantize/AssignMinEma/ReadVariableOp�Kquant_activation_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Mquant_activation_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�Equant_activation_6/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�@quant_activation_6/MovingAvgQuantize/AssignMaxEma/ReadVariableOp�Equant_activation_6/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�@quant_activation_6/MovingAvgQuantize/AssignMinEma/ReadVariableOp�Kquant_activation_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Mquant_activation_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�Equant_activation_7/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�@quant_activation_7/MovingAvgQuantize/AssignMaxEma/ReadVariableOp�Equant_activation_7/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�@quant_activation_7/MovingAvgQuantize/AssignMinEma/ReadVariableOp�Kquant_activation_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Mquant_activation_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�$quant_dense_5/BiasAdd/ReadVariableOp�#quant_dense_5/MatMul/ReadVariableOp�$quant_dense_6/BiasAdd/ReadVariableOp�#quant_dense_6/MatMul/ReadVariableOp�$quant_dense_7/BiasAdd/ReadVariableOp�#quant_dense_7/MatMul/ReadVariableOp�$quant_dense_8/BiasAdd/ReadVariableOp�#quant_dense_8/MatMul/ReadVariableOp�$quant_dense_9/BiasAdd/ReadVariableOp�*quant_dense_9/LastValueQuant/AssignMaxLast�*quant_dense_9/LastValueQuant/AssignMinLast�4quant_dense_9/LastValueQuant/BatchMax/ReadVariableOp�4quant_dense_9/LastValueQuant/BatchMin/ReadVariableOp�Cquant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Equant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Equant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�@quant_dense_9/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�;quant_dense_9/MovingAvgQuantize/AssignMaxEma/ReadVariableOp�@quant_dense_9/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�;quant_dense_9/MovingAvgQuantize/AssignMinEma/ReadVariableOp�Fquant_dense_9/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Hquant_dense_9/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�4quantize_layer_2/AllValuesQuantize/AssignMaxAllValue�4quantize_layer_2/AllValuesQuantize/AssignMinAllValue�Iquantize_layer_2/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Kquantize_layer_2/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�9quantize_layer_2/AllValuesQuantize/Maximum/ReadVariableOp�9quantize_layer_2/AllValuesQuantize/Minimum/ReadVariableOpy
(quantize_layer_2/AllValuesQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
+quantize_layer_2/AllValuesQuantize/BatchMinMininputs1quantize_layer_2/AllValuesQuantize/Const:output:0*
T0*
_output_shapes
: {
*quantize_layer_2/AllValuesQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
+quantize_layer_2/AllValuesQuantize/BatchMaxMaxinputs3quantize_layer_2/AllValuesQuantize/Const_1:output:0*
T0*
_output_shapes
: �
9quantize_layer_2/AllValuesQuantize/Minimum/ReadVariableOpReadVariableOpBquantize_layer_2_allvaluesquantize_minimum_readvariableop_resource*
_output_shapes
: *
dtype0�
*quantize_layer_2/AllValuesQuantize/MinimumMinimumAquantize_layer_2/AllValuesQuantize/Minimum/ReadVariableOp:value:04quantize_layer_2/AllValuesQuantize/BatchMin:output:0*
T0*
_output_shapes
: s
.quantize_layer_2/AllValuesQuantize/Minimum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,quantize_layer_2/AllValuesQuantize/Minimum_1Minimum.quantize_layer_2/AllValuesQuantize/Minimum:z:07quantize_layer_2/AllValuesQuantize/Minimum_1/y:output:0*
T0*
_output_shapes
: �
9quantize_layer_2/AllValuesQuantize/Maximum/ReadVariableOpReadVariableOpBquantize_layer_2_allvaluesquantize_maximum_readvariableop_resource*
_output_shapes
: *
dtype0�
*quantize_layer_2/AllValuesQuantize/MaximumMaximumAquantize_layer_2/AllValuesQuantize/Maximum/ReadVariableOp:value:04quantize_layer_2/AllValuesQuantize/BatchMax:output:0*
T0*
_output_shapes
: s
.quantize_layer_2/AllValuesQuantize/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,quantize_layer_2/AllValuesQuantize/Maximum_1Maximum.quantize_layer_2/AllValuesQuantize/Maximum:z:07quantize_layer_2/AllValuesQuantize/Maximum_1/y:output:0*
T0*
_output_shapes
: �
4quantize_layer_2/AllValuesQuantize/AssignMinAllValueAssignVariableOpBquantize_layer_2_allvaluesquantize_minimum_readvariableop_resource0quantize_layer_2/AllValuesQuantize/Minimum_1:z:0:^quantize_layer_2/AllValuesQuantize/Minimum/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
4quantize_layer_2/AllValuesQuantize/AssignMaxAllValueAssignVariableOpBquantize_layer_2_allvaluesquantize_maximum_readvariableop_resource0quantize_layer_2/AllValuesQuantize/Maximum_1:z:0:^quantize_layer_2/AllValuesQuantize/Maximum/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
Iquantize_layer_2/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpBquantize_layer_2_allvaluesquantize_minimum_readvariableop_resource5^quantize_layer_2/AllValuesQuantize/AssignMinAllValue*
_output_shapes
: *
dtype0�
Kquantize_layer_2/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpBquantize_layer_2_allvaluesquantize_maximum_readvariableop_resource5^quantize_layer_2/AllValuesQuantize/AssignMaxAllValue*
_output_shapes
: *
dtype0�
:quantize_layer_2/AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputsQquantize_layer_2/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Squantize_layer_2/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
#quant_dense_5/MatMul/ReadVariableOpReadVariableOp,quant_dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
quant_dense_5/MatMulMatMulDquantize_layer_2/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0+quant_dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
$quant_dense_5/BiasAdd/ReadVariableOpReadVariableOp-quant_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
quant_dense_5/BiasAddBiasAddquant_dense_5/MatMul:product:0,quant_dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������q
quant_activation_4/ReluReluquant_dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:���������{
*quant_activation_4/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
-quant_activation_4/MovingAvgQuantize/BatchMinMin%quant_activation_4/Relu:activations:03quant_activation_4/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: }
,quant_activation_4/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
-quant_activation_4/MovingAvgQuantize/BatchMaxMax%quant_activation_4/Relu:activations:05quant_activation_4/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: s
.quant_activation_4/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,quant_activation_4/MovingAvgQuantize/MinimumMinimum6quant_activation_4/MovingAvgQuantize/BatchMin:output:07quant_activation_4/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: s
.quant_activation_4/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,quant_activation_4/MovingAvgQuantize/MaximumMaximum6quant_activation_4/MovingAvgQuantize/BatchMax:output:07quant_activation_4/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: |
7quant_activation_4/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
@quant_activation_4/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpIquant_activation_4_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
5quant_activation_4/MovingAvgQuantize/AssignMinEma/subSubHquant_activation_4/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:00quant_activation_4/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
5quant_activation_4/MovingAvgQuantize/AssignMinEma/mulMul9quant_activation_4/MovingAvgQuantize/AssignMinEma/sub:z:0@quant_activation_4/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
Equant_activation_4/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpIquant_activation_4_movingavgquantize_assignminema_readvariableop_resource9quant_activation_4/MovingAvgQuantize/AssignMinEma/mul:z:0A^quant_activation_4/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0|
7quant_activation_4/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
@quant_activation_4/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpIquant_activation_4_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
5quant_activation_4/MovingAvgQuantize/AssignMaxEma/subSubHquant_activation_4/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:00quant_activation_4/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
5quant_activation_4/MovingAvgQuantize/AssignMaxEma/mulMul9quant_activation_4/MovingAvgQuantize/AssignMaxEma/sub:z:0@quant_activation_4/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
Equant_activation_4/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpIquant_activation_4_movingavgquantize_assignmaxema_readvariableop_resource9quant_activation_4/MovingAvgQuantize/AssignMaxEma/mul:z:0A^quant_activation_4/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0�
Kquant_activation_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpIquant_activation_4_movingavgquantize_assignminema_readvariableop_resourceF^quant_activation_4/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
Mquant_activation_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpIquant_activation_4_movingavgquantize_assignmaxema_readvariableop_resourceF^quant_activation_4/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
<quant_activation_4/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars%quant_activation_4/Relu:activations:0Squant_activation_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Uquant_activation_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
#quant_dense_6/MatMul/ReadVariableOpReadVariableOp,quant_dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
quant_dense_6/MatMulMatMulFquant_activation_4/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0+quant_dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
$quant_dense_6/BiasAdd/ReadVariableOpReadVariableOp-quant_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
quant_dense_6/BiasAddBiasAddquant_dense_6/MatMul:product:0,quant_dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������q
quant_activation_5/ReluReluquant_dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:���������{
*quant_activation_5/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
-quant_activation_5/MovingAvgQuantize/BatchMinMin%quant_activation_5/Relu:activations:03quant_activation_5/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: }
,quant_activation_5/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
-quant_activation_5/MovingAvgQuantize/BatchMaxMax%quant_activation_5/Relu:activations:05quant_activation_5/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: s
.quant_activation_5/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,quant_activation_5/MovingAvgQuantize/MinimumMinimum6quant_activation_5/MovingAvgQuantize/BatchMin:output:07quant_activation_5/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: s
.quant_activation_5/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,quant_activation_5/MovingAvgQuantize/MaximumMaximum6quant_activation_5/MovingAvgQuantize/BatchMax:output:07quant_activation_5/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: |
7quant_activation_5/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
@quant_activation_5/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpIquant_activation_5_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
5quant_activation_5/MovingAvgQuantize/AssignMinEma/subSubHquant_activation_5/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:00quant_activation_5/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
5quant_activation_5/MovingAvgQuantize/AssignMinEma/mulMul9quant_activation_5/MovingAvgQuantize/AssignMinEma/sub:z:0@quant_activation_5/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
Equant_activation_5/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpIquant_activation_5_movingavgquantize_assignminema_readvariableop_resource9quant_activation_5/MovingAvgQuantize/AssignMinEma/mul:z:0A^quant_activation_5/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0|
7quant_activation_5/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
@quant_activation_5/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpIquant_activation_5_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
5quant_activation_5/MovingAvgQuantize/AssignMaxEma/subSubHquant_activation_5/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:00quant_activation_5/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
5quant_activation_5/MovingAvgQuantize/AssignMaxEma/mulMul9quant_activation_5/MovingAvgQuantize/AssignMaxEma/sub:z:0@quant_activation_5/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
Equant_activation_5/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpIquant_activation_5_movingavgquantize_assignmaxema_readvariableop_resource9quant_activation_5/MovingAvgQuantize/AssignMaxEma/mul:z:0A^quant_activation_5/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0�
Kquant_activation_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpIquant_activation_5_movingavgquantize_assignminema_readvariableop_resourceF^quant_activation_5/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
Mquant_activation_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpIquant_activation_5_movingavgquantize_assignmaxema_readvariableop_resourceF^quant_activation_5/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
<quant_activation_5/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars%quant_activation_5/Relu:activations:0Squant_activation_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Uquant_activation_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
#quant_dense_7/MatMul/ReadVariableOpReadVariableOp,quant_dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
quant_dense_7/MatMulMatMulFquant_activation_5/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0+quant_dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
$quant_dense_7/BiasAdd/ReadVariableOpReadVariableOp-quant_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
quant_dense_7/BiasAddBiasAddquant_dense_7/MatMul:product:0,quant_dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������q
quant_activation_6/ReluReluquant_dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:���������{
*quant_activation_6/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
-quant_activation_6/MovingAvgQuantize/BatchMinMin%quant_activation_6/Relu:activations:03quant_activation_6/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: }
,quant_activation_6/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
-quant_activation_6/MovingAvgQuantize/BatchMaxMax%quant_activation_6/Relu:activations:05quant_activation_6/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: s
.quant_activation_6/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,quant_activation_6/MovingAvgQuantize/MinimumMinimum6quant_activation_6/MovingAvgQuantize/BatchMin:output:07quant_activation_6/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: s
.quant_activation_6/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,quant_activation_6/MovingAvgQuantize/MaximumMaximum6quant_activation_6/MovingAvgQuantize/BatchMax:output:07quant_activation_6/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: |
7quant_activation_6/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
@quant_activation_6/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpIquant_activation_6_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
5quant_activation_6/MovingAvgQuantize/AssignMinEma/subSubHquant_activation_6/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:00quant_activation_6/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
5quant_activation_6/MovingAvgQuantize/AssignMinEma/mulMul9quant_activation_6/MovingAvgQuantize/AssignMinEma/sub:z:0@quant_activation_6/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
Equant_activation_6/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpIquant_activation_6_movingavgquantize_assignminema_readvariableop_resource9quant_activation_6/MovingAvgQuantize/AssignMinEma/mul:z:0A^quant_activation_6/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0|
7quant_activation_6/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
@quant_activation_6/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpIquant_activation_6_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
5quant_activation_6/MovingAvgQuantize/AssignMaxEma/subSubHquant_activation_6/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:00quant_activation_6/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
5quant_activation_6/MovingAvgQuantize/AssignMaxEma/mulMul9quant_activation_6/MovingAvgQuantize/AssignMaxEma/sub:z:0@quant_activation_6/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
Equant_activation_6/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpIquant_activation_6_movingavgquantize_assignmaxema_readvariableop_resource9quant_activation_6/MovingAvgQuantize/AssignMaxEma/mul:z:0A^quant_activation_6/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0�
Kquant_activation_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpIquant_activation_6_movingavgquantize_assignminema_readvariableop_resourceF^quant_activation_6/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
Mquant_activation_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpIquant_activation_6_movingavgquantize_assignmaxema_readvariableop_resourceF^quant_activation_6/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
<quant_activation_6/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars%quant_activation_6/Relu:activations:0Squant_activation_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Uquant_activation_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
#quant_dense_8/MatMul/ReadVariableOpReadVariableOp,quant_dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
quant_dense_8/MatMulMatMulFquant_activation_6/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0+quant_dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
$quant_dense_8/BiasAdd/ReadVariableOpReadVariableOp-quant_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
quant_dense_8/BiasAddBiasAddquant_dense_8/MatMul:product:0,quant_dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������q
quant_activation_7/ReluReluquant_dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:���������{
*quant_activation_7/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
-quant_activation_7/MovingAvgQuantize/BatchMinMin%quant_activation_7/Relu:activations:03quant_activation_7/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: }
,quant_activation_7/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
-quant_activation_7/MovingAvgQuantize/BatchMaxMax%quant_activation_7/Relu:activations:05quant_activation_7/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: s
.quant_activation_7/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,quant_activation_7/MovingAvgQuantize/MinimumMinimum6quant_activation_7/MovingAvgQuantize/BatchMin:output:07quant_activation_7/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: s
.quant_activation_7/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,quant_activation_7/MovingAvgQuantize/MaximumMaximum6quant_activation_7/MovingAvgQuantize/BatchMax:output:07quant_activation_7/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: |
7quant_activation_7/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
@quant_activation_7/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpIquant_activation_7_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
5quant_activation_7/MovingAvgQuantize/AssignMinEma/subSubHquant_activation_7/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:00quant_activation_7/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
5quant_activation_7/MovingAvgQuantize/AssignMinEma/mulMul9quant_activation_7/MovingAvgQuantize/AssignMinEma/sub:z:0@quant_activation_7/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
Equant_activation_7/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpIquant_activation_7_movingavgquantize_assignminema_readvariableop_resource9quant_activation_7/MovingAvgQuantize/AssignMinEma/mul:z:0A^quant_activation_7/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0|
7quant_activation_7/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
@quant_activation_7/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpIquant_activation_7_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
5quant_activation_7/MovingAvgQuantize/AssignMaxEma/subSubHquant_activation_7/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:00quant_activation_7/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
5quant_activation_7/MovingAvgQuantize/AssignMaxEma/mulMul9quant_activation_7/MovingAvgQuantize/AssignMaxEma/sub:z:0@quant_activation_7/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
Equant_activation_7/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpIquant_activation_7_movingavgquantize_assignmaxema_readvariableop_resource9quant_activation_7/MovingAvgQuantize/AssignMaxEma/mul:z:0A^quant_activation_7/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0�
Kquant_activation_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpIquant_activation_7_movingavgquantize_assignminema_readvariableop_resourceF^quant_activation_7/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
Mquant_activation_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpIquant_activation_7_movingavgquantize_assignmaxema_readvariableop_resourceF^quant_activation_7/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
<quant_activation_7/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars%quant_activation_7/Relu:activations:0Squant_activation_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Uquant_activation_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������s
"quant_dense_9/LastValueQuant/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
4quant_dense_9/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp=quant_dense_9_lastvaluequant_batchmin_readvariableop_resource*
_output_shapes

:*
dtype0�
%quant_dense_9/LastValueQuant/BatchMinMin<quant_dense_9/LastValueQuant/BatchMin/ReadVariableOp:value:0+quant_dense_9/LastValueQuant/Const:output:0*
T0*
_output_shapes
: u
$quant_dense_9/LastValueQuant/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
4quant_dense_9/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp=quant_dense_9_lastvaluequant_batchmin_readvariableop_resource*
_output_shapes

:*
dtype0�
%quant_dense_9/LastValueQuant/BatchMaxMax<quant_dense_9/LastValueQuant/BatchMax/ReadVariableOp:value:0-quant_dense_9/LastValueQuant/Const_1:output:0*
T0*
_output_shapes
: k
&quant_dense_9/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
$quant_dense_9/LastValueQuant/truedivRealDiv.quant_dense_9/LastValueQuant/BatchMax:output:0/quant_dense_9/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: �
$quant_dense_9/LastValueQuant/MinimumMinimum.quant_dense_9/LastValueQuant/BatchMin:output:0(quant_dense_9/LastValueQuant/truediv:z:0*
T0*
_output_shapes
: g
"quant_dense_9/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
 quant_dense_9/LastValueQuant/mulMul.quant_dense_9/LastValueQuant/BatchMin:output:0+quant_dense_9/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: �
$quant_dense_9/LastValueQuant/MaximumMaximum.quant_dense_9/LastValueQuant/BatchMax:output:0$quant_dense_9/LastValueQuant/mul:z:0*
T0*
_output_shapes
: �
*quant_dense_9/LastValueQuant/AssignMinLastAssignVariableOp3quant_dense_9_lastvaluequant_assignminlast_resource(quant_dense_9/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(�
*quant_dense_9/LastValueQuant/AssignMaxLastAssignVariableOp3quant_dense_9_lastvaluequant_assignmaxlast_resource(quant_dense_9/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(�
Cquant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp=quant_dense_9_lastvaluequant_batchmin_readvariableop_resource*
_output_shapes

:*
dtype0�
Equant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp3quant_dense_9_lastvaluequant_assignminlast_resource+^quant_dense_9/LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0�
Equant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp3quant_dense_9_lastvaluequant_assignmaxlast_resource+^quant_dense_9/LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0�
4quant_dense_9/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsKquant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Mquant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Mquant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:*
narrow_range(�
quant_dense_9/MatMulMatMulFquant_activation_7/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0>quant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:����������
$quant_dense_9/BiasAdd/ReadVariableOpReadVariableOp-quant_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
quant_dense_9/BiasAddBiasAddquant_dense_9/MatMul:product:0,quant_dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������v
%quant_dense_9/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
(quant_dense_9/MovingAvgQuantize/BatchMinMinquant_dense_9/BiasAdd:output:0.quant_dense_9/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: x
'quant_dense_9/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
(quant_dense_9/MovingAvgQuantize/BatchMaxMaxquant_dense_9/BiasAdd:output:00quant_dense_9/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: n
)quant_dense_9/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
'quant_dense_9/MovingAvgQuantize/MinimumMinimum1quant_dense_9/MovingAvgQuantize/BatchMin:output:02quant_dense_9/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: n
)quant_dense_9/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
'quant_dense_9/MovingAvgQuantize/MaximumMaximum1quant_dense_9/MovingAvgQuantize/BatchMax:output:02quant_dense_9/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: w
2quant_dense_9/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
;quant_dense_9/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpDquant_dense_9_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
0quant_dense_9/MovingAvgQuantize/AssignMinEma/subSubCquant_dense_9/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0+quant_dense_9/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
0quant_dense_9/MovingAvgQuantize/AssignMinEma/mulMul4quant_dense_9/MovingAvgQuantize/AssignMinEma/sub:z:0;quant_dense_9/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
@quant_dense_9/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpDquant_dense_9_movingavgquantize_assignminema_readvariableop_resource4quant_dense_9/MovingAvgQuantize/AssignMinEma/mul:z:0<^quant_dense_9/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0w
2quant_dense_9/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
;quant_dense_9/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpDquant_dense_9_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
0quant_dense_9/MovingAvgQuantize/AssignMaxEma/subSubCquant_dense_9/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0+quant_dense_9/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
0quant_dense_9/MovingAvgQuantize/AssignMaxEma/mulMul4quant_dense_9/MovingAvgQuantize/AssignMaxEma/sub:z:0;quant_dense_9/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
@quant_dense_9/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpDquant_dense_9_movingavgquantize_assignmaxema_readvariableop_resource4quant_dense_9/MovingAvgQuantize/AssignMaxEma/mul:z:0<^quant_dense_9/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0�
Fquant_dense_9/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpDquant_dense_9_movingavgquantize_assignminema_readvariableop_resourceA^quant_dense_9/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
Hquant_dense_9/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpDquant_dense_9_movingavgquantize_assignmaxema_readvariableop_resourceA^quant_dense_9/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
7quant_dense_9/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsquant_dense_9/BiasAdd:output:0Nquant_dense_9/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Pquant_dense_9/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
quant_dense_9/SoftmaxSoftmaxAquant_dense_9/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������n
IdentityIdentityquant_dense_9/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOpF^quant_activation_4/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpA^quant_activation_4/MovingAvgQuantize/AssignMaxEma/ReadVariableOpF^quant_activation_4/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpA^quant_activation_4/MovingAvgQuantize/AssignMinEma/ReadVariableOpL^quant_activation_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpN^quant_activation_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1F^quant_activation_5/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpA^quant_activation_5/MovingAvgQuantize/AssignMaxEma/ReadVariableOpF^quant_activation_5/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpA^quant_activation_5/MovingAvgQuantize/AssignMinEma/ReadVariableOpL^quant_activation_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpN^quant_activation_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1F^quant_activation_6/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpA^quant_activation_6/MovingAvgQuantize/AssignMaxEma/ReadVariableOpF^quant_activation_6/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpA^quant_activation_6/MovingAvgQuantize/AssignMinEma/ReadVariableOpL^quant_activation_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpN^quant_activation_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1F^quant_activation_7/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpA^quant_activation_7/MovingAvgQuantize/AssignMaxEma/ReadVariableOpF^quant_activation_7/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpA^quant_activation_7/MovingAvgQuantize/AssignMinEma/ReadVariableOpL^quant_activation_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpN^quant_activation_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1%^quant_dense_5/BiasAdd/ReadVariableOp$^quant_dense_5/MatMul/ReadVariableOp%^quant_dense_6/BiasAdd/ReadVariableOp$^quant_dense_6/MatMul/ReadVariableOp%^quant_dense_7/BiasAdd/ReadVariableOp$^quant_dense_7/MatMul/ReadVariableOp%^quant_dense_8/BiasAdd/ReadVariableOp$^quant_dense_8/MatMul/ReadVariableOp%^quant_dense_9/BiasAdd/ReadVariableOp+^quant_dense_9/LastValueQuant/AssignMaxLast+^quant_dense_9/LastValueQuant/AssignMinLast5^quant_dense_9/LastValueQuant/BatchMax/ReadVariableOp5^quant_dense_9/LastValueQuant/BatchMin/ReadVariableOpD^quant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpF^quant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1F^quant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2A^quant_dense_9/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp<^quant_dense_9/MovingAvgQuantize/AssignMaxEma/ReadVariableOpA^quant_dense_9/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp<^quant_dense_9/MovingAvgQuantize/AssignMinEma/ReadVariableOpG^quant_dense_9/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpI^quant_dense_9/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_15^quantize_layer_2/AllValuesQuantize/AssignMaxAllValue5^quantize_layer_2/AllValuesQuantize/AssignMinAllValueJ^quantize_layer_2/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpL^quantize_layer_2/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:^quantize_layer_2/AllValuesQuantize/Maximum/ReadVariableOp:^quantize_layer_2/AllValuesQuantize/Minimum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������: : : : : : : : : : : : : : : : : : : : : : : : 2�
Equant_activation_4/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpEquant_activation_4/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2�
@quant_activation_4/MovingAvgQuantize/AssignMaxEma/ReadVariableOp@quant_activation_4/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2�
Equant_activation_4/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpEquant_activation_4/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2�
@quant_activation_4/MovingAvgQuantize/AssignMinEma/ReadVariableOp@quant_activation_4/MovingAvgQuantize/AssignMinEma/ReadVariableOp2�
Kquant_activation_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpKquant_activation_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Mquant_activation_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Mquant_activation_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Equant_activation_5/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpEquant_activation_5/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2�
@quant_activation_5/MovingAvgQuantize/AssignMaxEma/ReadVariableOp@quant_activation_5/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2�
Equant_activation_5/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpEquant_activation_5/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2�
@quant_activation_5/MovingAvgQuantize/AssignMinEma/ReadVariableOp@quant_activation_5/MovingAvgQuantize/AssignMinEma/ReadVariableOp2�
Kquant_activation_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpKquant_activation_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Mquant_activation_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Mquant_activation_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Equant_activation_6/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpEquant_activation_6/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2�
@quant_activation_6/MovingAvgQuantize/AssignMaxEma/ReadVariableOp@quant_activation_6/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2�
Equant_activation_6/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpEquant_activation_6/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2�
@quant_activation_6/MovingAvgQuantize/AssignMinEma/ReadVariableOp@quant_activation_6/MovingAvgQuantize/AssignMinEma/ReadVariableOp2�
Kquant_activation_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpKquant_activation_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Mquant_activation_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Mquant_activation_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Equant_activation_7/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpEquant_activation_7/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2�
@quant_activation_7/MovingAvgQuantize/AssignMaxEma/ReadVariableOp@quant_activation_7/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2�
Equant_activation_7/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpEquant_activation_7/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2�
@quant_activation_7/MovingAvgQuantize/AssignMinEma/ReadVariableOp@quant_activation_7/MovingAvgQuantize/AssignMinEma/ReadVariableOp2�
Kquant_activation_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpKquant_activation_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Mquant_activation_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Mquant_activation_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12L
$quant_dense_5/BiasAdd/ReadVariableOp$quant_dense_5/BiasAdd/ReadVariableOp2J
#quant_dense_5/MatMul/ReadVariableOp#quant_dense_5/MatMul/ReadVariableOp2L
$quant_dense_6/BiasAdd/ReadVariableOp$quant_dense_6/BiasAdd/ReadVariableOp2J
#quant_dense_6/MatMul/ReadVariableOp#quant_dense_6/MatMul/ReadVariableOp2L
$quant_dense_7/BiasAdd/ReadVariableOp$quant_dense_7/BiasAdd/ReadVariableOp2J
#quant_dense_7/MatMul/ReadVariableOp#quant_dense_7/MatMul/ReadVariableOp2L
$quant_dense_8/BiasAdd/ReadVariableOp$quant_dense_8/BiasAdd/ReadVariableOp2J
#quant_dense_8/MatMul/ReadVariableOp#quant_dense_8/MatMul/ReadVariableOp2L
$quant_dense_9/BiasAdd/ReadVariableOp$quant_dense_9/BiasAdd/ReadVariableOp2X
*quant_dense_9/LastValueQuant/AssignMaxLast*quant_dense_9/LastValueQuant/AssignMaxLast2X
*quant_dense_9/LastValueQuant/AssignMinLast*quant_dense_9/LastValueQuant/AssignMinLast2l
4quant_dense_9/LastValueQuant/BatchMax/ReadVariableOp4quant_dense_9/LastValueQuant/BatchMax/ReadVariableOp2l
4quant_dense_9/LastValueQuant/BatchMin/ReadVariableOp4quant_dense_9/LastValueQuant/BatchMin/ReadVariableOp2�
Cquant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpCquant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Equant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Equant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Equant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Equant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
@quant_dense_9/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp@quant_dense_9/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2z
;quant_dense_9/MovingAvgQuantize/AssignMaxEma/ReadVariableOp;quant_dense_9/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2�
@quant_dense_9/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp@quant_dense_9/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2z
;quant_dense_9/MovingAvgQuantize/AssignMinEma/ReadVariableOp;quant_dense_9/MovingAvgQuantize/AssignMinEma/ReadVariableOp2�
Fquant_dense_9/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpFquant_dense_9/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Hquant_dense_9/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Hquant_dense_9/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12l
4quantize_layer_2/AllValuesQuantize/AssignMaxAllValue4quantize_layer_2/AllValuesQuantize/AssignMaxAllValue2l
4quantize_layer_2/AllValuesQuantize/AssignMinAllValue4quantize_layer_2/AllValuesQuantize/AssignMinAllValue2�
Iquantize_layer_2/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpIquantize_layer_2/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Kquantize_layer_2/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Kquantize_layer_2/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12v
9quantize_layer_2/AllValuesQuantize/Maximum/ReadVariableOp9quantize_layer_2/AllValuesQuantize/Maximum/ReadVariableOp2v
9quantize_layer_2/AllValuesQuantize/Minimum/ReadVariableOp9quantize_layer_2/AllValuesQuantize/Minimum/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�;
�

H__inference_sequential_1_layer_call_and_return_conditional_losses_174038
dense_5_input!
quantize_layer_2_173979: !
quantize_layer_2_173981: &
quant_dense_5_173984:"
quant_dense_5_173986:#
quant_activation_4_173989: #
quant_activation_4_173991: &
quant_dense_6_173994:"
quant_dense_6_173996:#
quant_activation_5_173999: #
quant_activation_5_174001: &
quant_dense_7_174004:"
quant_dense_7_174006:#
quant_activation_6_174009: #
quant_activation_6_174011: &
quant_dense_8_174014:"
quant_dense_8_174016:#
quant_activation_7_174019: #
quant_activation_7_174021: &
quant_dense_9_174024:
quant_dense_9_174026: 
quant_dense_9_174028: "
quant_dense_9_174030:
quant_dense_9_174032: 
quant_dense_9_174034: 
identity��*quant_activation_4/StatefulPartitionedCall�*quant_activation_5/StatefulPartitionedCall�*quant_activation_6/StatefulPartitionedCall�*quant_activation_7/StatefulPartitionedCall�%quant_dense_5/StatefulPartitionedCall�%quant_dense_6/StatefulPartitionedCall�%quant_dense_7/StatefulPartitionedCall�%quant_dense_8/StatefulPartitionedCall�%quant_dense_9/StatefulPartitionedCall�(quantize_layer_2/StatefulPartitionedCall�
(quantize_layer_2/StatefulPartitionedCallStatefulPartitionedCalldense_5_inputquantize_layer_2_173979quantize_layer_2_173981*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_quantize_layer_2_layer_call_and_return_conditional_losses_173686�
%quant_dense_5/StatefulPartitionedCallStatefulPartitionedCall1quantize_layer_2/StatefulPartitionedCall:output:0quant_dense_5_173984quant_dense_5_173986*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_quant_dense_5_layer_call_and_return_conditional_losses_173646�
*quant_activation_4/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_5/StatefulPartitionedCall:output:0quant_activation_4_173989quant_activation_4_173991*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_quant_activation_4_layer_call_and_return_conditional_losses_173617�
%quant_dense_6/StatefulPartitionedCallStatefulPartitionedCall3quant_activation_4/StatefulPartitionedCall:output:0quant_dense_6_173994quant_dense_6_173996*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_quant_dense_6_layer_call_and_return_conditional_losses_173572�
*quant_activation_5/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_6/StatefulPartitionedCall:output:0quant_activation_5_173999quant_activation_5_174001*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_quant_activation_5_layer_call_and_return_conditional_losses_173543�
%quant_dense_7/StatefulPartitionedCallStatefulPartitionedCall3quant_activation_5/StatefulPartitionedCall:output:0quant_dense_7_174004quant_dense_7_174006*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_quant_dense_7_layer_call_and_return_conditional_losses_173498�
*quant_activation_6/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_7/StatefulPartitionedCall:output:0quant_activation_6_174009quant_activation_6_174011*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_quant_activation_6_layer_call_and_return_conditional_losses_173469�
%quant_dense_8/StatefulPartitionedCallStatefulPartitionedCall3quant_activation_6/StatefulPartitionedCall:output:0quant_dense_8_174014quant_dense_8_174016*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_quant_dense_8_layer_call_and_return_conditional_losses_173424�
*quant_activation_7/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_8/StatefulPartitionedCall:output:0quant_activation_7_174019quant_activation_7_174021*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_quant_activation_7_layer_call_and_return_conditional_losses_173395�
%quant_dense_9/StatefulPartitionedCallStatefulPartitionedCall3quant_activation_7/StatefulPartitionedCall:output:0quant_dense_9_174024quant_dense_9_174026quant_dense_9_174028quant_dense_9_174030quant_dense_9_174032quant_dense_9_174034*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_quant_dense_9_layer_call_and_return_conditional_losses_173342}
IdentityIdentity.quant_dense_9/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp+^quant_activation_4/StatefulPartitionedCall+^quant_activation_5/StatefulPartitionedCall+^quant_activation_6/StatefulPartitionedCall+^quant_activation_7/StatefulPartitionedCall&^quant_dense_5/StatefulPartitionedCall&^quant_dense_6/StatefulPartitionedCall&^quant_dense_7/StatefulPartitionedCall&^quant_dense_8/StatefulPartitionedCall&^quant_dense_9/StatefulPartitionedCall)^quantize_layer_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������: : : : : : : : : : : : : : : : : : : : : : : : 2X
*quant_activation_4/StatefulPartitionedCall*quant_activation_4/StatefulPartitionedCall2X
*quant_activation_5/StatefulPartitionedCall*quant_activation_5/StatefulPartitionedCall2X
*quant_activation_6/StatefulPartitionedCall*quant_activation_6/StatefulPartitionedCall2X
*quant_activation_7/StatefulPartitionedCall*quant_activation_7/StatefulPartitionedCall2N
%quant_dense_5/StatefulPartitionedCall%quant_dense_5/StatefulPartitionedCall2N
%quant_dense_6/StatefulPartitionedCall%quant_dense_6/StatefulPartitionedCall2N
%quant_dense_7/StatefulPartitionedCall%quant_dense_7/StatefulPartitionedCall2N
%quant_dense_8/StatefulPartitionedCall%quant_dense_8/StatefulPartitionedCall2N
%quant_dense_9/StatefulPartitionedCall%quant_dense_9/StatefulPartitionedCall2T
(quantize_layer_2/StatefulPartitionedCall(quantize_layer_2/StatefulPartitionedCall:V R
'
_output_shapes
:���������
'
_user_specified_namedense_5_input
�;
�

H__inference_sequential_1_layer_call_and_return_conditional_losses_173222

inputs!
quantize_layer_2_173053: !
quantize_layer_2_173055: &
quant_dense_5_173069:"
quant_dense_5_173071:#
quant_activation_4_173085: #
quant_activation_4_173087: &
quant_dense_6_173101:"
quant_dense_6_173103:#
quant_activation_5_173117: #
quant_activation_5_173119: &
quant_dense_7_173133:"
quant_dense_7_173135:#
quant_activation_6_173149: #
quant_activation_6_173151: &
quant_dense_8_173165:"
quant_dense_8_173167:#
quant_activation_7_173181: #
quant_activation_7_173183: &
quant_dense_9_173208:
quant_dense_9_173210: 
quant_dense_9_173212: "
quant_dense_9_173214:
quant_dense_9_173216: 
quant_dense_9_173218: 
identity��*quant_activation_4/StatefulPartitionedCall�*quant_activation_5/StatefulPartitionedCall�*quant_activation_6/StatefulPartitionedCall�*quant_activation_7/StatefulPartitionedCall�%quant_dense_5/StatefulPartitionedCall�%quant_dense_6/StatefulPartitionedCall�%quant_dense_7/StatefulPartitionedCall�%quant_dense_8/StatefulPartitionedCall�%quant_dense_9/StatefulPartitionedCall�(quantize_layer_2/StatefulPartitionedCall�
(quantize_layer_2/StatefulPartitionedCallStatefulPartitionedCallinputsquantize_layer_2_173053quantize_layer_2_173055*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_quantize_layer_2_layer_call_and_return_conditional_losses_173052�
%quant_dense_5/StatefulPartitionedCallStatefulPartitionedCall1quantize_layer_2/StatefulPartitionedCall:output:0quant_dense_5_173069quant_dense_5_173071*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_quant_dense_5_layer_call_and_return_conditional_losses_173068�
*quant_activation_4/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_5/StatefulPartitionedCall:output:0quant_activation_4_173085quant_activation_4_173087*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_quant_activation_4_layer_call_and_return_conditional_losses_173084�
%quant_dense_6/StatefulPartitionedCallStatefulPartitionedCall3quant_activation_4/StatefulPartitionedCall:output:0quant_dense_6_173101quant_dense_6_173103*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_quant_dense_6_layer_call_and_return_conditional_losses_173100�
*quant_activation_5/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_6/StatefulPartitionedCall:output:0quant_activation_5_173117quant_activation_5_173119*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_quant_activation_5_layer_call_and_return_conditional_losses_173116�
%quant_dense_7/StatefulPartitionedCallStatefulPartitionedCall3quant_activation_5/StatefulPartitionedCall:output:0quant_dense_7_173133quant_dense_7_173135*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_quant_dense_7_layer_call_and_return_conditional_losses_173132�
*quant_activation_6/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_7/StatefulPartitionedCall:output:0quant_activation_6_173149quant_activation_6_173151*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_quant_activation_6_layer_call_and_return_conditional_losses_173148�
%quant_dense_8/StatefulPartitionedCallStatefulPartitionedCall3quant_activation_6/StatefulPartitionedCall:output:0quant_dense_8_173165quant_dense_8_173167*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_quant_dense_8_layer_call_and_return_conditional_losses_173164�
*quant_activation_7/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_8/StatefulPartitionedCall:output:0quant_activation_7_173181quant_activation_7_173183*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_quant_activation_7_layer_call_and_return_conditional_losses_173180�
%quant_dense_9/StatefulPartitionedCallStatefulPartitionedCall3quant_activation_7/StatefulPartitionedCall:output:0quant_dense_9_173208quant_dense_9_173210quant_dense_9_173212quant_dense_9_173214quant_dense_9_173216quant_dense_9_173218*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_quant_dense_9_layer_call_and_return_conditional_losses_173207}
IdentityIdentity.quant_dense_9/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp+^quant_activation_4/StatefulPartitionedCall+^quant_activation_5/StatefulPartitionedCall+^quant_activation_6/StatefulPartitionedCall+^quant_activation_7/StatefulPartitionedCall&^quant_dense_5/StatefulPartitionedCall&^quant_dense_6/StatefulPartitionedCall&^quant_dense_7/StatefulPartitionedCall&^quant_dense_8/StatefulPartitionedCall&^quant_dense_9/StatefulPartitionedCall)^quantize_layer_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������: : : : : : : : : : : : : : : : : : : : : : : : 2X
*quant_activation_4/StatefulPartitionedCall*quant_activation_4/StatefulPartitionedCall2X
*quant_activation_5/StatefulPartitionedCall*quant_activation_5/StatefulPartitionedCall2X
*quant_activation_6/StatefulPartitionedCall*quant_activation_6/StatefulPartitionedCall2X
*quant_activation_7/StatefulPartitionedCall*quant_activation_7/StatefulPartitionedCall2N
%quant_dense_5/StatefulPartitionedCall%quant_dense_5/StatefulPartitionedCall2N
%quant_dense_6/StatefulPartitionedCall%quant_dense_6/StatefulPartitionedCall2N
%quant_dense_7/StatefulPartitionedCall%quant_dense_7/StatefulPartitionedCall2N
%quant_dense_8/StatefulPartitionedCall%quant_dense_8/StatefulPartitionedCall2N
%quant_dense_9/StatefulPartitionedCall%quant_dense_9/StatefulPartitionedCall2T
(quantize_layer_2/StatefulPartitionedCall(quantize_layer_2/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�)
�
N__inference_quant_activation_6_layer_call_and_return_conditional_losses_174781

inputs@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1F
ReluReluinputs*
T0*'
_output_shapes
:���������h
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: j
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0�
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 2h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
3__inference_quant_activation_7_layer_call_fn_174837

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_quant_activation_7_layer_call_and_return_conditional_losses_173395o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
I__inference_quant_dense_8_layer_call_and_return_conditional_losses_174809

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
-__inference_sequential_1_layer_call_fn_174205

inputs
unknown: 
	unknown_0: 
	unknown_1:
	unknown_2:
	unknown_3: 
	unknown_4: 
	unknown_5:
	unknown_6:
	unknown_7: 
	unknown_8: 
	unknown_9:

unknown_10:

unknown_11: 

unknown_12: 

unknown_13:

unknown_14:

unknown_15: 

unknown_16: 

unknown_17:

unknown_18: 

unknown_19: 

unknown_20:

unknown_21: 

unknown_22: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_173810o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
I__inference_quant_dense_6_layer_call_and_return_conditional_losses_174625

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
.__inference_quant_dense_6_layer_call_fn_174606

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_quant_dense_6_layer_call_and_return_conditional_losses_173100o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
I__inference_quant_dense_8_layer_call_and_return_conditional_losses_173424

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
N__inference_quant_activation_5_layer_call_and_return_conditional_losses_174663

inputsK
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1F
ReluReluinputs*
T0*'
_output_shapes
:����������
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
I__inference_quant_dense_7_layer_call_and_return_conditional_losses_173132

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
.__inference_quant_dense_7_layer_call_fn_174707

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_quant_dense_7_layer_call_and_return_conditional_losses_173498o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
1__inference_quantize_layer_2_layer_call_fn_174466

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_quantize_layer_2_layer_call_and_return_conditional_losses_173052o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
.__inference_quant_dense_9_layer_call_fn_174890

inputs
unknown:
	unknown_0: 
	unknown_1: 
	unknown_2:
	unknown_3: 
	unknown_4: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_quant_dense_9_layer_call_and_return_conditional_losses_173207o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�)
�
N__inference_quant_activation_7_layer_call_and_return_conditional_losses_173395

inputs@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1F
ReluReluinputs*
T0*'
_output_shapes
:���������h
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: j
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0�
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 2h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
3__inference_quant_activation_5_layer_call_fn_174653

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_quant_activation_5_layer_call_and_return_conditional_losses_173543o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
I__inference_quant_dense_7_layer_call_and_return_conditional_losses_174717

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�&
"__inference__traced_restore_175388
file_prefix@
6assignvariableop_quantize_layer_2_quantize_layer_2_min: B
8assignvariableop_1_quantize_layer_2_quantize_layer_2_max: <
2assignvariableop_2_quantize_layer_2_optimizer_step: 9
/assignvariableop_3_quant_dense_5_optimizer_step: >
4assignvariableop_4_quant_activation_4_optimizer_step: :
0assignvariableop_5_quant_activation_4_output_min: :
0assignvariableop_6_quant_activation_4_output_max: 9
/assignvariableop_7_quant_dense_6_optimizer_step: >
4assignvariableop_8_quant_activation_5_optimizer_step: :
0assignvariableop_9_quant_activation_5_output_min: ;
1assignvariableop_10_quant_activation_5_output_max: :
0assignvariableop_11_quant_dense_7_optimizer_step: ?
5assignvariableop_12_quant_activation_6_optimizer_step: ;
1assignvariableop_13_quant_activation_6_output_min: ;
1assignvariableop_14_quant_activation_6_output_max: :
0assignvariableop_15_quant_dense_8_optimizer_step: ?
5assignvariableop_16_quant_activation_7_optimizer_step: ;
1assignvariableop_17_quant_activation_7_output_min: ;
1assignvariableop_18_quant_activation_7_output_max: :
0assignvariableop_19_quant_dense_9_optimizer_step: 6
,assignvariableop_20_quant_dense_9_kernel_min: 6
,assignvariableop_21_quant_dense_9_kernel_max: >
4assignvariableop_22_quant_dense_9_pre_activation_min: >
4assignvariableop_23_quant_dense_9_pre_activation_max: 4
"assignvariableop_24_dense_5_kernel:.
 assignvariableop_25_dense_5_bias:4
"assignvariableop_26_dense_6_kernel:.
 assignvariableop_27_dense_6_bias:4
"assignvariableop_28_dense_7_kernel:.
 assignvariableop_29_dense_7_bias:4
"assignvariableop_30_dense_8_kernel:.
 assignvariableop_31_dense_8_bias:4
"assignvariableop_32_dense_9_kernel:.
 assignvariableop_33_dense_9_bias:'
assignvariableop_34_adam_iter:	 )
assignvariableop_35_adam_beta_1: )
assignvariableop_36_adam_beta_2: (
assignvariableop_37_adam_decay: 0
&assignvariableop_38_adam_learning_rate: %
assignvariableop_39_total_1: %
assignvariableop_40_count_1: #
assignvariableop_41_total: #
assignvariableop_42_count: ;
)assignvariableop_43_adam_dense_5_kernel_m:5
'assignvariableop_44_adam_dense_5_bias_m:;
)assignvariableop_45_adam_dense_6_kernel_m:5
'assignvariableop_46_adam_dense_6_bias_m:;
)assignvariableop_47_adam_dense_7_kernel_m:5
'assignvariableop_48_adam_dense_7_bias_m:;
)assignvariableop_49_adam_dense_8_kernel_m:5
'assignvariableop_50_adam_dense_8_bias_m:;
)assignvariableop_51_adam_dense_9_kernel_m:5
'assignvariableop_52_adam_dense_9_bias_m:;
)assignvariableop_53_adam_dense_5_kernel_v:5
'assignvariableop_54_adam_dense_5_bias_v:;
)assignvariableop_55_adam_dense_6_kernel_v:5
'assignvariableop_56_adam_dense_6_bias_v:;
)assignvariableop_57_adam_dense_7_kernel_v:5
'assignvariableop_58_adam_dense_7_bias_v:;
)assignvariableop_59_adam_dense_8_kernel_v:5
'assignvariableop_60_adam_dense_8_bias_v:;
)assignvariableop_61_adam_dense_9_kernel_v:5
'assignvariableop_62_adam_dense_9_bias_v:
identity_64��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*�
value�B�@BDlayer_with_weights-0/quantize_layer_2_min/.ATTRIBUTES/VARIABLE_VALUEBDlayer_with_weights-0/quantize_layer_2_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-0/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-1/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-2/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/output_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/output_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-3/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-4/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/output_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/output_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-5/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-6/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-6/output_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-6/output_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-7/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-8/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-8/output_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-8/output_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-9/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-9/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-9/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-9/pre_activation_min/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-9/pre_activation_max/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*�
value�B�@B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*N
dtypesD
B2@	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp6assignvariableop_quantize_layer_2_quantize_layer_2_minIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp8assignvariableop_1_quantize_layer_2_quantize_layer_2_maxIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp2assignvariableop_2_quantize_layer_2_optimizer_stepIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp/assignvariableop_3_quant_dense_5_optimizer_stepIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp4assignvariableop_4_quant_activation_4_optimizer_stepIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp0assignvariableop_5_quant_activation_4_output_minIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp0assignvariableop_6_quant_activation_4_output_maxIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp/assignvariableop_7_quant_dense_6_optimizer_stepIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp4assignvariableop_8_quant_activation_5_optimizer_stepIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp0assignvariableop_9_quant_activation_5_output_minIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp1assignvariableop_10_quant_activation_5_output_maxIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp0assignvariableop_11_quant_dense_7_optimizer_stepIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp5assignvariableop_12_quant_activation_6_optimizer_stepIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp1assignvariableop_13_quant_activation_6_output_minIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp1assignvariableop_14_quant_activation_6_output_maxIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp0assignvariableop_15_quant_dense_8_optimizer_stepIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp5assignvariableop_16_quant_activation_7_optimizer_stepIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp1assignvariableop_17_quant_activation_7_output_minIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp1assignvariableop_18_quant_activation_7_output_maxIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp0assignvariableop_19_quant_dense_9_optimizer_stepIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp,assignvariableop_20_quant_dense_9_kernel_minIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp,assignvariableop_21_quant_dense_9_kernel_maxIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp4assignvariableop_22_quant_dense_9_pre_activation_minIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp4assignvariableop_23_quant_dense_9_pre_activation_maxIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp"assignvariableop_24_dense_5_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp assignvariableop_25_dense_5_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp"assignvariableop_26_dense_6_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp assignvariableop_27_dense_6_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp"assignvariableop_28_dense_7_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp assignvariableop_29_dense_7_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp"assignvariableop_30_dense_8_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp assignvariableop_31_dense_8_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp"assignvariableop_32_dense_9_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp assignvariableop_33_dense_9_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_34AssignVariableOpassignvariableop_34_adam_iterIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOpassignvariableop_35_adam_beta_1Identity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOpassignvariableop_36_adam_beta_2Identity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOpassignvariableop_37_adam_decayIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp&assignvariableop_38_adam_learning_rateIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOpassignvariableop_39_total_1Identity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOpassignvariableop_40_count_1Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOpassignvariableop_41_totalIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOpassignvariableop_42_countIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_dense_5_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp'assignvariableop_44_adam_dense_5_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp)assignvariableop_45_adam_dense_6_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp'assignvariableop_46_adam_dense_6_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp)assignvariableop_47_adam_dense_7_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp'assignvariableop_48_adam_dense_7_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp)assignvariableop_49_adam_dense_8_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp'assignvariableop_50_adam_dense_8_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp)assignvariableop_51_adam_dense_9_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp'assignvariableop_52_adam_dense_9_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp)assignvariableop_53_adam_dense_5_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp'assignvariableop_54_adam_dense_5_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp)assignvariableop_55_adam_dense_6_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp'assignvariableop_56_adam_dense_6_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp)assignvariableop_57_adam_dense_7_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp'assignvariableop_58_adam_dense_7_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp)assignvariableop_59_adam_dense_8_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOp'assignvariableop_60_adam_dense_8_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOp)assignvariableop_61_adam_dense_9_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOp'assignvariableop_62_adam_dense_9_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_63Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_64IdentityIdentity_63:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_64Identity_64:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�	
�
I__inference_quant_dense_7_layer_call_and_return_conditional_losses_173498

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
-__inference_sequential_1_layer_call_fn_174152

inputs
unknown: 
	unknown_0: 
	unknown_1:
	unknown_2:
	unknown_3: 
	unknown_4: 
	unknown_5:
	unknown_6:
	unknown_7: 
	unknown_8: 
	unknown_9:

unknown_10:

unknown_11: 

unknown_12: 

unknown_13:

unknown_14:

unknown_15: 

unknown_16: 

unknown_17:

unknown_18: 

unknown_19: 

unknown_20:

unknown_21: 

unknown_22: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_173222o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
N__inference_quant_activation_6_layer_call_and_return_conditional_losses_173148

inputsK
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1F
ReluReluinputs*
T0*'
_output_shapes
:����������
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�N
�
I__inference_quant_dense_9_layer_call_and_return_conditional_losses_173342

inputsA
/lastvaluequant_batchmin_readvariableop_resource:/
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�LastValueQuant/AssignMaxLast�LastValueQuant/AssignMinLast�&LastValueQuant/BatchMax/ReadVariableOp�&LastValueQuant/BatchMin/ReadVariableOp�5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1e
LastValueQuant/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*
_output_shapes

:*
dtype0�
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/Const:output:0*
T0*
_output_shapes
: g
LastValueQuant/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*
_output_shapes

:*
dtype0�
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:0LastValueQuant/Const_1:output:0*
T0*
_output_shapes
: ]
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: �
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
: Y
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��{
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: |
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
: �
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype0*
validate_shape(�
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype0*
validate_shape(�
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*
_output_shapes

:*
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype0�
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype0�
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:*
narrow_range(|
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       v
MovingAvgQuantize/BatchMinMinBiasAdd:output:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: j
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       x
MovingAvgQuantize/BatchMaxMaxBiasAdd:output:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0�
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsBiasAdd:output:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:���������y
SoftmaxSoftmax3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�)
�
N__inference_quant_activation_6_layer_call_and_return_conditional_losses_173469

inputs@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity��2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMaxEma/ReadVariableOp�2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp�-MovingAvgQuantize/AssignMinEma/ReadVariableOp�8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1F
ReluReluinputs*
T0*'
_output_shapes
:���������h
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       x
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: j
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       z
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: `
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: i
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype0i
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype0�
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: �
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: �
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype0�
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype0�
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 2h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
3__inference_quant_activation_6_layer_call_fn_174745

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_quant_activation_6_layer_call_and_return_conditional_losses_173469o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�"
�
L__inference_quantize_layer_2_layer_call_and_return_conditional_losses_174505

inputs;
1allvaluesquantize_minimum_readvariableop_resource: ;
1allvaluesquantize_maximum_readvariableop_resource: 
identity��#AllValuesQuantize/AssignMaxAllValue�#AllValuesQuantize/AssignMinAllValue�8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�(AllValuesQuantize/Maximum/ReadVariableOp�(AllValuesQuantize/Minimum/ReadVariableOph
AllValuesQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       l
AllValuesQuantize/BatchMinMininputs AllValuesQuantize/Const:output:0*
T0*
_output_shapes
: j
AllValuesQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       n
AllValuesQuantize/BatchMaxMaxinputs"AllValuesQuantize/Const_1:output:0*
T0*
_output_shapes
: �
(AllValuesQuantize/Minimum/ReadVariableOpReadVariableOp1allvaluesquantize_minimum_readvariableop_resource*
_output_shapes
: *
dtype0�
AllValuesQuantize/MinimumMinimum0AllValuesQuantize/Minimum/ReadVariableOp:value:0#AllValuesQuantize/BatchMin:output:0*
T0*
_output_shapes
: b
AllValuesQuantize/Minimum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
AllValuesQuantize/Minimum_1MinimumAllValuesQuantize/Minimum:z:0&AllValuesQuantize/Minimum_1/y:output:0*
T0*
_output_shapes
: �
(AllValuesQuantize/Maximum/ReadVariableOpReadVariableOp1allvaluesquantize_maximum_readvariableop_resource*
_output_shapes
: *
dtype0�
AllValuesQuantize/MaximumMaximum0AllValuesQuantize/Maximum/ReadVariableOp:value:0#AllValuesQuantize/BatchMax:output:0*
T0*
_output_shapes
: b
AllValuesQuantize/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
AllValuesQuantize/Maximum_1MaximumAllValuesQuantize/Maximum:z:0&AllValuesQuantize/Maximum_1/y:output:0*
T0*
_output_shapes
: �
#AllValuesQuantize/AssignMinAllValueAssignVariableOp1allvaluesquantize_minimum_readvariableop_resourceAllValuesQuantize/Minimum_1:z:0)^AllValuesQuantize/Minimum/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
#AllValuesQuantize/AssignMaxAllValueAssignVariableOp1allvaluesquantize_maximum_readvariableop_resourceAllValuesQuantize/Maximum_1:z:0)^AllValuesQuantize/Maximum/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp1allvaluesquantize_minimum_readvariableop_resource$^AllValuesQuantize/AssignMinAllValue*
_output_shapes
: *
dtype0�
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp1allvaluesquantize_maximum_readvariableop_resource$^AllValuesQuantize/AssignMaxAllValue*
_output_shapes
: *
dtype0�
)AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputs@AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BAllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
IdentityIdentity3AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp$^AllValuesQuantize/AssignMaxAllValue$^AllValuesQuantize/AssignMinAllValue9^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1)^AllValuesQuantize/Maximum/ReadVariableOp)^AllValuesQuantize/Minimum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 2J
#AllValuesQuantize/AssignMaxAllValue#AllValuesQuantize/AssignMaxAllValue2J
#AllValuesQuantize/AssignMinAllValue#AllValuesQuantize/AssignMinAllValue2t
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12T
(AllValuesQuantize/Maximum/ReadVariableOp(AllValuesQuantize/Maximum/ReadVariableOp2T
(AllValuesQuantize/Minimum/ReadVariableOp(AllValuesQuantize/Minimum/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
I__inference_quant_dense_8_layer_call_and_return_conditional_losses_173164

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�u
�
__inference__traced_save_175189
file_prefixD
@savev2_quantize_layer_2_quantize_layer_2_min_read_readvariableopD
@savev2_quantize_layer_2_quantize_layer_2_max_read_readvariableop>
:savev2_quantize_layer_2_optimizer_step_read_readvariableop;
7savev2_quant_dense_5_optimizer_step_read_readvariableop@
<savev2_quant_activation_4_optimizer_step_read_readvariableop<
8savev2_quant_activation_4_output_min_read_readvariableop<
8savev2_quant_activation_4_output_max_read_readvariableop;
7savev2_quant_dense_6_optimizer_step_read_readvariableop@
<savev2_quant_activation_5_optimizer_step_read_readvariableop<
8savev2_quant_activation_5_output_min_read_readvariableop<
8savev2_quant_activation_5_output_max_read_readvariableop;
7savev2_quant_dense_7_optimizer_step_read_readvariableop@
<savev2_quant_activation_6_optimizer_step_read_readvariableop<
8savev2_quant_activation_6_output_min_read_readvariableop<
8savev2_quant_activation_6_output_max_read_readvariableop;
7savev2_quant_dense_8_optimizer_step_read_readvariableop@
<savev2_quant_activation_7_optimizer_step_read_readvariableop<
8savev2_quant_activation_7_output_min_read_readvariableop<
8savev2_quant_activation_7_output_max_read_readvariableop;
7savev2_quant_dense_9_optimizer_step_read_readvariableop7
3savev2_quant_dense_9_kernel_min_read_readvariableop7
3savev2_quant_dense_9_kernel_max_read_readvariableop?
;savev2_quant_dense_9_pre_activation_min_read_readvariableop?
;savev2_quant_dense_9_pre_activation_max_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop4
0savev2_adam_dense_5_kernel_m_read_readvariableop2
.savev2_adam_dense_5_bias_m_read_readvariableop4
0savev2_adam_dense_6_kernel_m_read_readvariableop2
.savev2_adam_dense_6_bias_m_read_readvariableop4
0savev2_adam_dense_7_kernel_m_read_readvariableop2
.savev2_adam_dense_7_bias_m_read_readvariableop4
0savev2_adam_dense_8_kernel_m_read_readvariableop2
.savev2_adam_dense_8_bias_m_read_readvariableop4
0savev2_adam_dense_9_kernel_m_read_readvariableop2
.savev2_adam_dense_9_bias_m_read_readvariableop4
0savev2_adam_dense_5_kernel_v_read_readvariableop2
.savev2_adam_dense_5_bias_v_read_readvariableop4
0savev2_adam_dense_6_kernel_v_read_readvariableop2
.savev2_adam_dense_6_bias_v_read_readvariableop4
0savev2_adam_dense_7_kernel_v_read_readvariableop2
.savev2_adam_dense_7_bias_v_read_readvariableop4
0savev2_adam_dense_8_kernel_v_read_readvariableop2
.savev2_adam_dense_8_bias_v_read_readvariableop4
0savev2_adam_dense_9_kernel_v_read_readvariableop2
.savev2_adam_dense_9_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*�
value�B�@BDlayer_with_weights-0/quantize_layer_2_min/.ATTRIBUTES/VARIABLE_VALUEBDlayer_with_weights-0/quantize_layer_2_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-0/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-1/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-2/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/output_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/output_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-3/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-4/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/output_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/output_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-5/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-6/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-6/output_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-6/output_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-7/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-8/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-8/output_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-8/output_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-9/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-9/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-9/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-9/pre_activation_min/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-9/pre_activation_max/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*�
value�B�@B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0@savev2_quantize_layer_2_quantize_layer_2_min_read_readvariableop@savev2_quantize_layer_2_quantize_layer_2_max_read_readvariableop:savev2_quantize_layer_2_optimizer_step_read_readvariableop7savev2_quant_dense_5_optimizer_step_read_readvariableop<savev2_quant_activation_4_optimizer_step_read_readvariableop8savev2_quant_activation_4_output_min_read_readvariableop8savev2_quant_activation_4_output_max_read_readvariableop7savev2_quant_dense_6_optimizer_step_read_readvariableop<savev2_quant_activation_5_optimizer_step_read_readvariableop8savev2_quant_activation_5_output_min_read_readvariableop8savev2_quant_activation_5_output_max_read_readvariableop7savev2_quant_dense_7_optimizer_step_read_readvariableop<savev2_quant_activation_6_optimizer_step_read_readvariableop8savev2_quant_activation_6_output_min_read_readvariableop8savev2_quant_activation_6_output_max_read_readvariableop7savev2_quant_dense_8_optimizer_step_read_readvariableop<savev2_quant_activation_7_optimizer_step_read_readvariableop8savev2_quant_activation_7_output_min_read_readvariableop8savev2_quant_activation_7_output_max_read_readvariableop7savev2_quant_dense_9_optimizer_step_read_readvariableop3savev2_quant_dense_9_kernel_min_read_readvariableop3savev2_quant_dense_9_kernel_max_read_readvariableop;savev2_quant_dense_9_pre_activation_min_read_readvariableop;savev2_quant_dense_9_pre_activation_max_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop0savev2_adam_dense_5_kernel_m_read_readvariableop.savev2_adam_dense_5_bias_m_read_readvariableop0savev2_adam_dense_6_kernel_m_read_readvariableop.savev2_adam_dense_6_bias_m_read_readvariableop0savev2_adam_dense_7_kernel_m_read_readvariableop.savev2_adam_dense_7_bias_m_read_readvariableop0savev2_adam_dense_8_kernel_m_read_readvariableop.savev2_adam_dense_8_bias_m_read_readvariableop0savev2_adam_dense_9_kernel_m_read_readvariableop.savev2_adam_dense_9_bias_m_read_readvariableop0savev2_adam_dense_5_kernel_v_read_readvariableop.savev2_adam_dense_5_bias_v_read_readvariableop0savev2_adam_dense_6_kernel_v_read_readvariableop.savev2_adam_dense_6_bias_v_read_readvariableop0savev2_adam_dense_7_kernel_v_read_readvariableop.savev2_adam_dense_7_bias_v_read_readvariableop0savev2_adam_dense_8_kernel_v_read_readvariableop.savev2_adam_dense_8_bias_v_read_readvariableop0savev2_adam_dense_9_kernel_v_read_readvariableop.savev2_adam_dense_9_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *N
dtypesD
B2@	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : ::::::::::: : : : : : : : : ::::::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

::  

_output_shapes
::$! 

_output_shapes

:: "

_output_shapes
::#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :$, 

_output_shapes

:: -

_output_shapes
::$. 

_output_shapes

:: /

_output_shapes
::$0 

_output_shapes

:: 1

_output_shapes
::$2 

_output_shapes

:: 3

_output_shapes
::$4 

_output_shapes

:: 5

_output_shapes
::$6 

_output_shapes

:: 7

_output_shapes
::$8 

_output_shapes

:: 9

_output_shapes
::$: 

_output_shapes

:: ;

_output_shapes
::$< 

_output_shapes

:: =

_output_shapes
::$> 

_output_shapes

:: ?

_output_shapes
::@

_output_shapes
: 
ۋ
�
H__inference_sequential_1_layer_call_and_return_conditional_losses_174279

inputs\
Rquantize_layer_2_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource: ^
Tquantize_layer_2_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource: >
,quant_dense_5_matmul_readvariableop_resource:;
-quant_dense_5_biasadd_readvariableop_resource:^
Tquant_activation_4_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: `
Vquant_activation_4_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: >
,quant_dense_6_matmul_readvariableop_resource:;
-quant_dense_6_biasadd_readvariableop_resource:^
Tquant_activation_5_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: `
Vquant_activation_5_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: >
,quant_dense_7_matmul_readvariableop_resource:;
-quant_dense_7_biasadd_readvariableop_resource:^
Tquant_activation_6_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: `
Vquant_activation_6_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: >
,quant_dense_8_matmul_readvariableop_resource:;
-quant_dense_8_biasadd_readvariableop_resource:^
Tquant_activation_7_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: `
Vquant_activation_7_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: ^
Lquant_dense_9_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:X
Nquant_dense_9_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: X
Nquant_dense_9_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: ;
-quant_dense_9_biasadd_readvariableop_resource:Y
Oquant_dense_9_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: [
Qquant_dense_9_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity��Kquant_activation_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Mquant_activation_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�Kquant_activation_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Mquant_activation_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�Kquant_activation_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Mquant_activation_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�Kquant_activation_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Mquant_activation_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�$quant_dense_5/BiasAdd/ReadVariableOp�#quant_dense_5/MatMul/ReadVariableOp�$quant_dense_6/BiasAdd/ReadVariableOp�#quant_dense_6/MatMul/ReadVariableOp�$quant_dense_7/BiasAdd/ReadVariableOp�#quant_dense_7/MatMul/ReadVariableOp�$quant_dense_8/BiasAdd/ReadVariableOp�#quant_dense_8/MatMul/ReadVariableOp�$quant_dense_9/BiasAdd/ReadVariableOp�Cquant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp�Equant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1�Equant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2�Fquant_dense_9/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Hquant_dense_9/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�Iquantize_layer_2/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp�Kquantize_layer_2/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1�
Iquantize_layer_2/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpRquantize_layer_2_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Kquantize_layer_2/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpTquantize_layer_2_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
:quantize_layer_2/AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputsQquantize_layer_2/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Squantize_layer_2/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
#quant_dense_5/MatMul/ReadVariableOpReadVariableOp,quant_dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
quant_dense_5/MatMulMatMulDquantize_layer_2/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0+quant_dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
$quant_dense_5/BiasAdd/ReadVariableOpReadVariableOp-quant_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
quant_dense_5/BiasAddBiasAddquant_dense_5/MatMul:product:0,quant_dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������q
quant_activation_4/ReluReluquant_dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:����������
Kquant_activation_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpTquant_activation_4_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Mquant_activation_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpVquant_activation_4_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
<quant_activation_4/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars%quant_activation_4/Relu:activations:0Squant_activation_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Uquant_activation_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
#quant_dense_6/MatMul/ReadVariableOpReadVariableOp,quant_dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
quant_dense_6/MatMulMatMulFquant_activation_4/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0+quant_dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
$quant_dense_6/BiasAdd/ReadVariableOpReadVariableOp-quant_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
quant_dense_6/BiasAddBiasAddquant_dense_6/MatMul:product:0,quant_dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������q
quant_activation_5/ReluReluquant_dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:����������
Kquant_activation_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpTquant_activation_5_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Mquant_activation_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpVquant_activation_5_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
<quant_activation_5/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars%quant_activation_5/Relu:activations:0Squant_activation_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Uquant_activation_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
#quant_dense_7/MatMul/ReadVariableOpReadVariableOp,quant_dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
quant_dense_7/MatMulMatMulFquant_activation_5/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0+quant_dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
$quant_dense_7/BiasAdd/ReadVariableOpReadVariableOp-quant_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
quant_dense_7/BiasAddBiasAddquant_dense_7/MatMul:product:0,quant_dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������q
quant_activation_6/ReluReluquant_dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:����������
Kquant_activation_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpTquant_activation_6_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Mquant_activation_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpVquant_activation_6_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
<quant_activation_6/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars%quant_activation_6/Relu:activations:0Squant_activation_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Uquant_activation_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
#quant_dense_8/MatMul/ReadVariableOpReadVariableOp,quant_dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
quant_dense_8/MatMulMatMulFquant_activation_6/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0+quant_dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
$quant_dense_8/BiasAdd/ReadVariableOpReadVariableOp-quant_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
quant_dense_8/BiasAddBiasAddquant_dense_8/MatMul:product:0,quant_dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������q
quant_activation_7/ReluReluquant_dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:����������
Kquant_activation_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpTquant_activation_7_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Mquant_activation_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpVquant_activation_7_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
<quant_activation_7/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars%quant_activation_7/Relu:activations:0Squant_activation_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Uquant_activation_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
Cquant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpLquant_dense_9_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:*
dtype0�
Equant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpNquant_dense_9_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Equant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpNquant_dense_9_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype0�
4quant_dense_9/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsKquant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Mquant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Mquant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:*
narrow_range(�
quant_dense_9/MatMulMatMulFquant_activation_7/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0>quant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:����������
$quant_dense_9/BiasAdd/ReadVariableOpReadVariableOp-quant_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
quant_dense_9/BiasAddBiasAddquant_dense_9/MatMul:product:0,quant_dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
Fquant_dense_9/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpOquant_dense_9_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype0�
Hquant_dense_9/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpQquant_dense_9_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype0�
7quant_dense_9/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsquant_dense_9/BiasAdd:output:0Nquant_dense_9/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Pquant_dense_9/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:����������
quant_dense_9/SoftmaxSoftmaxAquant_dense_9/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:���������n
IdentityIdentityquant_dense_9/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOpL^quant_activation_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpN^quant_activation_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1L^quant_activation_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpN^quant_activation_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1L^quant_activation_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpN^quant_activation_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1L^quant_activation_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpN^quant_activation_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1%^quant_dense_5/BiasAdd/ReadVariableOp$^quant_dense_5/MatMul/ReadVariableOp%^quant_dense_6/BiasAdd/ReadVariableOp$^quant_dense_6/MatMul/ReadVariableOp%^quant_dense_7/BiasAdd/ReadVariableOp$^quant_dense_7/MatMul/ReadVariableOp%^quant_dense_8/BiasAdd/ReadVariableOp$^quant_dense_8/MatMul/ReadVariableOp%^quant_dense_9/BiasAdd/ReadVariableOpD^quant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpF^quant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1F^quant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2G^quant_dense_9/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpI^quant_dense_9/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1J^quantize_layer_2/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpL^quantize_layer_2/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������: : : : : : : : : : : : : : : : : : : : : : : : 2�
Kquant_activation_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpKquant_activation_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Mquant_activation_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Mquant_activation_4/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Kquant_activation_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpKquant_activation_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Mquant_activation_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Mquant_activation_5/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Kquant_activation_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpKquant_activation_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Mquant_activation_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Mquant_activation_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Kquant_activation_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpKquant_activation_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Mquant_activation_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Mquant_activation_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12L
$quant_dense_5/BiasAdd/ReadVariableOp$quant_dense_5/BiasAdd/ReadVariableOp2J
#quant_dense_5/MatMul/ReadVariableOp#quant_dense_5/MatMul/ReadVariableOp2L
$quant_dense_6/BiasAdd/ReadVariableOp$quant_dense_6/BiasAdd/ReadVariableOp2J
#quant_dense_6/MatMul/ReadVariableOp#quant_dense_6/MatMul/ReadVariableOp2L
$quant_dense_7/BiasAdd/ReadVariableOp$quant_dense_7/BiasAdd/ReadVariableOp2J
#quant_dense_7/MatMul/ReadVariableOp#quant_dense_7/MatMul/ReadVariableOp2L
$quant_dense_8/BiasAdd/ReadVariableOp$quant_dense_8/BiasAdd/ReadVariableOp2J
#quant_dense_8/MatMul/ReadVariableOp#quant_dense_8/MatMul/ReadVariableOp2L
$quant_dense_9/BiasAdd/ReadVariableOp$quant_dense_9/BiasAdd/ReadVariableOp2�
Cquant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpCquant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2�
Equant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Equant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Equant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Equant_dense_9/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22�
Fquant_dense_9/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpFquant_dense_9/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Hquant_dense_9/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Hquant_dense_9/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12�
Iquantize_layer_2/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpIquantize_layer_2/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2�
Kquantize_layer_2/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Kquantize_layer_2/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
3__inference_quant_activation_5_layer_call_fn_174644

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_quant_activation_5_layer_call_and_return_conditional_losses_173116o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
3__inference_quant_activation_6_layer_call_fn_174736

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_quant_activation_6_layer_call_and_return_conditional_losses_173148o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
I__inference_quant_dense_6_layer_call_and_return_conditional_losses_173572

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
.__inference_quant_dense_7_layer_call_fn_174698

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_quant_dense_7_layer_call_and_return_conditional_losses_173132o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs"�	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
G
dense_5_input6
serving_default_dense_5_input:0���������A
quant_dense_90
StatefulPartitionedCall:0���������tensorflow/serving/predict:˶
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
layer_with_weights-6
layer-6
layer_with_weights-7
layer-7
	layer_with_weights-8
	layer-8

layer_with_weights-9

layer-9
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
quantize_layer_2_min
quantize_layer_2_max
quantizer_vars
optimizer_step"
_tf_keras_layer
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses
	$layer
%optimizer_step
&_weight_vars
'_quantize_activations
(_output_quantizers"
_tf_keras_layer
�
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses
	/layer
0optimizer_step
1_weight_vars
2_quantize_activations
3_output_quantizers
4
output_min
5
output_max
6_output_quantizer_vars"
_tf_keras_layer
�
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses
	=layer
>optimizer_step
?_weight_vars
@_quantize_activations
A_output_quantizers"
_tf_keras_layer
�
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses
	Hlayer
Ioptimizer_step
J_weight_vars
K_quantize_activations
L_output_quantizers
M
output_min
N
output_max
O_output_quantizer_vars"
_tf_keras_layer
�
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses
	Vlayer
Woptimizer_step
X_weight_vars
Y_quantize_activations
Z_output_quantizers"
_tf_keras_layer
�
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses
	alayer
boptimizer_step
c_weight_vars
d_quantize_activations
e_output_quantizers
f
output_min
g
output_max
h_output_quantizer_vars"
_tf_keras_layer
�
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses
	olayer
poptimizer_step
q_weight_vars
r_quantize_activations
s_output_quantizers"
_tf_keras_layer
�
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses
	zlayer
{optimizer_step
|_weight_vars
}_quantize_activations
~_output_quantizers

output_min
�
output_max
�_output_quantizer_vars"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

�layer
�optimizer_step
�_weight_vars
�
kernel_min
�
kernel_max
�_quantize_activations
�pre_activation_min
�pre_activation_max
�_output_quantizers"
_tf_keras_layer
�
0
1
2
�3
�4
%5
06
47
58
�9
�10
>11
I12
M13
N14
�15
�16
W17
b18
f19
g20
�21
�22
p23
{24
25
�26
�27
�28
�29
�30
�31
�32
�33"
trackable_list_wrapper
p
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
-__inference_sequential_1_layer_call_fn_173273
-__inference_sequential_1_layer_call_fn_174152
-__inference_sequential_1_layer_call_fn_174205
-__inference_sequential_1_layer_call_fn_173914�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
H__inference_sequential_1_layer_call_and_return_conditional_losses_174279
H__inference_sequential_1_layer_call_and_return_conditional_losses_174457
H__inference_sequential_1_layer_call_and_return_conditional_losses_173976
H__inference_sequential_1_layer_call_and_return_conditional_losses_174038�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�B�
!__inference__wrapped_model_173036dense_5_input"�
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
�
	�iter
�beta_1
�beta_2

�decay
�learning_rate	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�"
	optimizer
-
�serving_default"
signature_map
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
1__inference_quantize_layer_2_layer_call_fn_174466
1__inference_quantize_layer_2_layer_call_fn_174475�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
L__inference_quantize_layer_2_layer_call_and_return_conditional_losses_174484
L__inference_quantize_layer_2_layer_call_and_return_conditional_losses_174505�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
-:+ 2%quantize_layer_2/quantize_layer_2_min
-:+ 2%quantize_layer_2/quantize_layer_2_max
:
min_var
max_var"
trackable_dict_wrapper
':% 2quantize_layer_2/optimizer_step
7
�0
�1
%2"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
.__inference_quant_dense_5_layer_call_fn_174514
.__inference_quant_dense_5_layer_call_fn_174523�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
I__inference_quant_dense_5_layer_call_and_return_conditional_losses_174533
I__inference_quant_dense_5_layer_call_and_return_conditional_losses_174543�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
$:" 2quant_dense_5/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
00
41
52"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
3__inference_quant_activation_4_layer_call_fn_174552
3__inference_quant_activation_4_layer_call_fn_174561�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
N__inference_quant_activation_4_layer_call_and_return_conditional_losses_174571
N__inference_quant_activation_4_layer_call_and_return_conditional_losses_174597�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
):' 2!quant_activation_4/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
%:# 2quant_activation_4/output_min
%:# 2quant_activation_4/output_max
:
4min_var
5max_var"
trackable_dict_wrapper
7
�0
�1
>2"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
.__inference_quant_dense_6_layer_call_fn_174606
.__inference_quant_dense_6_layer_call_fn_174615�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
I__inference_quant_dense_6_layer_call_and_return_conditional_losses_174625
I__inference_quant_dense_6_layer_call_and_return_conditional_losses_174635�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
$:" 2quant_dense_6/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
I0
M1
N2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
3__inference_quant_activation_5_layer_call_fn_174644
3__inference_quant_activation_5_layer_call_fn_174653�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
N__inference_quant_activation_5_layer_call_and_return_conditional_losses_174663
N__inference_quant_activation_5_layer_call_and_return_conditional_losses_174689�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
):' 2!quant_activation_5/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
%:# 2quant_activation_5/output_min
%:# 2quant_activation_5/output_max
:
Mmin_var
Nmax_var"
trackable_dict_wrapper
7
�0
�1
W2"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
.__inference_quant_dense_7_layer_call_fn_174698
.__inference_quant_dense_7_layer_call_fn_174707�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
I__inference_quant_dense_7_layer_call_and_return_conditional_losses_174717
I__inference_quant_dense_7_layer_call_and_return_conditional_losses_174727�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
$:" 2quant_dense_7/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
b0
f1
g2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
3__inference_quant_activation_6_layer_call_fn_174736
3__inference_quant_activation_6_layer_call_fn_174745�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
N__inference_quant_activation_6_layer_call_and_return_conditional_losses_174755
N__inference_quant_activation_6_layer_call_and_return_conditional_losses_174781�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
):' 2!quant_activation_6/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
%:# 2quant_activation_6/output_min
%:# 2quant_activation_6/output_max
:
fmin_var
gmax_var"
trackable_dict_wrapper
7
�0
�1
p2"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
.__inference_quant_dense_8_layer_call_fn_174790
.__inference_quant_dense_8_layer_call_fn_174799�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
I__inference_quant_dense_8_layer_call_and_return_conditional_losses_174809
I__inference_quant_dense_8_layer_call_and_return_conditional_losses_174819�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
$:" 2quant_dense_8/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
6
{0
1
�2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
3__inference_quant_activation_7_layer_call_fn_174828
3__inference_quant_activation_7_layer_call_fn_174837�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
N__inference_quant_activation_7_layer_call_and_return_conditional_losses_174847
N__inference_quant_activation_7_layer_call_and_return_conditional_losses_174873�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
):' 2!quant_activation_7/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
%:# 2quant_activation_7/output_min
%:# 2quant_activation_7/output_max
;
min_var
�max_var"
trackable_dict_wrapper
X
�0
�1
�2
�3
�4
�5
�6"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
.__inference_quant_dense_9_layer_call_fn_174890
.__inference_quant_dense_9_layer_call_fn_174907�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
I__inference_quant_dense_9_layer_call_and_return_conditional_losses_174928
I__inference_quant_dense_9_layer_call_and_return_conditional_losses_174977�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
$:" 2quant_dense_9/optimizer_step
(
�0"
trackable_list_wrapper
 : 2quant_dense_9/kernel_min
 : 2quant_dense_9/kernel_max
 "
trackable_list_wrapper
(:& 2 quant_dense_9/pre_activation_min
(:& 2 quant_dense_9/pre_activation_max
 "
trackable_list_wrapper
 :2dense_5/kernel
:2dense_5/bias
 :2dense_6/kernel
:2dense_6/bias
 :2dense_7/kernel
:2dense_7/bias
 :2dense_8/kernel
:2dense_8/bias
 :2dense_9/kernel
:2dense_9/bias
�
0
1
2
%3
04
45
56
>7
I8
M9
N10
W11
b12
f13
g14
p15
{16
17
�18
�19
�20
�21
�22
�23"
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
	8

9"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_sequential_1_layer_call_fn_173273dense_5_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_sequential_1_layer_call_fn_174152inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_sequential_1_layer_call_fn_174205inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_sequential_1_layer_call_fn_173914dense_5_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_sequential_1_layer_call_and_return_conditional_losses_174279inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_sequential_1_layer_call_and_return_conditional_losses_174457inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_sequential_1_layer_call_and_return_conditional_losses_173976dense_5_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_sequential_1_layer_call_and_return_conditional_losses_174038dense_5_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
�B�
$__inference_signature_wrapper_174099dense_5_input"�
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
5
0
1
2"
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
1__inference_quantize_layer_2_layer_call_fn_174466inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
1__inference_quantize_layer_2_layer_call_fn_174475inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_quantize_layer_2_layer_call_and_return_conditional_losses_174484inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_quantize_layer_2_layer_call_and_return_conditional_losses_174505inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
'
%0"
trackable_list_wrapper
'
$0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_quant_dense_5_layer_call_fn_174514inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_quant_dense_5_layer_call_fn_174523inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_quant_dense_5_layer_call_and_return_conditional_losses_174533inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_quant_dense_5_layer_call_and_return_conditional_losses_174543inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
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
�2��
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
5
00
41
52"
trackable_list_wrapper
'
/0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
3__inference_quant_activation_4_layer_call_fn_174552inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
3__inference_quant_activation_4_layer_call_fn_174561inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
N__inference_quant_activation_4_layer_call_and_return_conditional_losses_174571inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
N__inference_quant_activation_4_layer_call_and_return_conditional_losses_174597inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
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
�2��
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
'
>0"
trackable_list_wrapper
'
=0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_quant_dense_6_layer_call_fn_174606inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_quant_dense_6_layer_call_fn_174615inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_quant_dense_6_layer_call_and_return_conditional_losses_174625inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_quant_dense_6_layer_call_and_return_conditional_losses_174635inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
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
�2��
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
5
I0
M1
N2"
trackable_list_wrapper
'
H0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
3__inference_quant_activation_5_layer_call_fn_174644inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
3__inference_quant_activation_5_layer_call_fn_174653inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
N__inference_quant_activation_5_layer_call_and_return_conditional_losses_174663inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
N__inference_quant_activation_5_layer_call_and_return_conditional_losses_174689inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
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
�2��
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
'
W0"
trackable_list_wrapper
'
V0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_quant_dense_7_layer_call_fn_174698inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_quant_dense_7_layer_call_fn_174707inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_quant_dense_7_layer_call_and_return_conditional_losses_174717inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_quant_dense_7_layer_call_and_return_conditional_losses_174727inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
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
�2��
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
5
b0
f1
g2"
trackable_list_wrapper
'
a0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
3__inference_quant_activation_6_layer_call_fn_174736inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
3__inference_quant_activation_6_layer_call_fn_174745inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
N__inference_quant_activation_6_layer_call_and_return_conditional_losses_174755inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
N__inference_quant_activation_6_layer_call_and_return_conditional_losses_174781inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
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
�2��
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
'
p0"
trackable_list_wrapper
'
o0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_quant_dense_8_layer_call_fn_174790inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_quant_dense_8_layer_call_fn_174799inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_quant_dense_8_layer_call_and_return_conditional_losses_174809inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_quant_dense_8_layer_call_and_return_conditional_losses_174819inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
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
�2��
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
6
{0
1
�2"
trackable_list_wrapper
'
z0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
3__inference_quant_activation_7_layer_call_fn_174828inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
3__inference_quant_activation_7_layer_call_fn_174837inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
N__inference_quant_activation_7_layer_call_and_return_conditional_losses_174847inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
N__inference_quant_activation_7_layer_call_and_return_conditional_losses_174873inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
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
�2��
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
H
�0
�1
�2
�3
�4"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_quant_dense_9_layer_call_fn_174890inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_quant_dense_9_layer_call_fn_174907inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_quant_dense_9_layer_call_and_return_conditional_losses_174928inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_quant_dense_9_layer_call_and_return_conditional_losses_174977inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
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
�2��
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
1
�0
�2"
trackable_tuple_wrapper
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
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
<
�min_var
�max_var"
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
%:#2Adam/dense_5/kernel/m
:2Adam/dense_5/bias/m
%:#2Adam/dense_6/kernel/m
:2Adam/dense_6/bias/m
%:#2Adam/dense_7/kernel/m
:2Adam/dense_7/bias/m
%:#2Adam/dense_8/kernel/m
:2Adam/dense_8/bias/m
%:#2Adam/dense_9/kernel/m
:2Adam/dense_9/bias/m
%:#2Adam/dense_5/kernel/v
:2Adam/dense_5/bias/v
%:#2Adam/dense_6/kernel/v
:2Adam/dense_6/bias/v
%:#2Adam/dense_7/kernel/v
:2Adam/dense_7/bias/v
%:#2Adam/dense_8/kernel/v
:2Adam/dense_8/bias/v
%:#2Adam/dense_9/kernel/v
:2Adam/dense_9/bias/v�
!__inference__wrapped_model_173036�'��45��MN��fg���������6�3
,�)
'�$
dense_5_input���������
� "=�:
8
quant_dense_9'�$
quant_dense_9����������
N__inference_quant_activation_4_layer_call_and_return_conditional_losses_174571`453�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
N__inference_quant_activation_4_layer_call_and_return_conditional_losses_174597`453�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
3__inference_quant_activation_4_layer_call_fn_174552S453�0
)�&
 �
inputs���������
p 
� "�����������
3__inference_quant_activation_4_layer_call_fn_174561S453�0
)�&
 �
inputs���������
p
� "�����������
N__inference_quant_activation_5_layer_call_and_return_conditional_losses_174663`MN3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
N__inference_quant_activation_5_layer_call_and_return_conditional_losses_174689`MN3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
3__inference_quant_activation_5_layer_call_fn_174644SMN3�0
)�&
 �
inputs���������
p 
� "�����������
3__inference_quant_activation_5_layer_call_fn_174653SMN3�0
)�&
 �
inputs���������
p
� "�����������
N__inference_quant_activation_6_layer_call_and_return_conditional_losses_174755`fg3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
N__inference_quant_activation_6_layer_call_and_return_conditional_losses_174781`fg3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
3__inference_quant_activation_6_layer_call_fn_174736Sfg3�0
)�&
 �
inputs���������
p 
� "�����������
3__inference_quant_activation_6_layer_call_fn_174745Sfg3�0
)�&
 �
inputs���������
p
� "�����������
N__inference_quant_activation_7_layer_call_and_return_conditional_losses_174847a�3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
N__inference_quant_activation_7_layer_call_and_return_conditional_losses_174873a�3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
3__inference_quant_activation_7_layer_call_fn_174828T�3�0
)�&
 �
inputs���������
p 
� "�����������
3__inference_quant_activation_7_layer_call_fn_174837T�3�0
)�&
 �
inputs���������
p
� "�����������
I__inference_quant_dense_5_layer_call_and_return_conditional_losses_174533b��3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
I__inference_quant_dense_5_layer_call_and_return_conditional_losses_174543b��3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
.__inference_quant_dense_5_layer_call_fn_174514U��3�0
)�&
 �
inputs���������
p 
� "�����������
.__inference_quant_dense_5_layer_call_fn_174523U��3�0
)�&
 �
inputs���������
p
� "�����������
I__inference_quant_dense_6_layer_call_and_return_conditional_losses_174625b��3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
I__inference_quant_dense_6_layer_call_and_return_conditional_losses_174635b��3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
.__inference_quant_dense_6_layer_call_fn_174606U��3�0
)�&
 �
inputs���������
p 
� "�����������
.__inference_quant_dense_6_layer_call_fn_174615U��3�0
)�&
 �
inputs���������
p
� "�����������
I__inference_quant_dense_7_layer_call_and_return_conditional_losses_174717b��3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
I__inference_quant_dense_7_layer_call_and_return_conditional_losses_174727b��3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
.__inference_quant_dense_7_layer_call_fn_174698U��3�0
)�&
 �
inputs���������
p 
� "�����������
.__inference_quant_dense_7_layer_call_fn_174707U��3�0
)�&
 �
inputs���������
p
� "�����������
I__inference_quant_dense_8_layer_call_and_return_conditional_losses_174809b��3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
I__inference_quant_dense_8_layer_call_and_return_conditional_losses_174819b��3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
.__inference_quant_dense_8_layer_call_fn_174790U��3�0
)�&
 �
inputs���������
p 
� "�����������
.__inference_quant_dense_8_layer_call_fn_174799U��3�0
)�&
 �
inputs���������
p
� "�����������
I__inference_quant_dense_9_layer_call_and_return_conditional_losses_174928j������3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
I__inference_quant_dense_9_layer_call_and_return_conditional_losses_174977j������3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
.__inference_quant_dense_9_layer_call_fn_174890]������3�0
)�&
 �
inputs���������
p 
� "�����������
.__inference_quant_dense_9_layer_call_fn_174907]������3�0
)�&
 �
inputs���������
p
� "�����������
L__inference_quantize_layer_2_layer_call_and_return_conditional_losses_174484`3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
L__inference_quantize_layer_2_layer_call_and_return_conditional_losses_174505`3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
1__inference_quantize_layer_2_layer_call_fn_174466S3�0
)�&
 �
inputs���������
p 
� "�����������
1__inference_quantize_layer_2_layer_call_fn_174475S3�0
)�&
 �
inputs���������
p
� "�����������
H__inference_sequential_1_layer_call_and_return_conditional_losses_173976�'��45��MN��fg���������>�;
4�1
'�$
dense_5_input���������
p 

 
� "%�"
�
0���������
� �
H__inference_sequential_1_layer_call_and_return_conditional_losses_174038�'��45��MN��fg���������>�;
4�1
'�$
dense_5_input���������
p

 
� "%�"
�
0���������
� �
H__inference_sequential_1_layer_call_and_return_conditional_losses_174279�'��45��MN��fg���������7�4
-�*
 �
inputs���������
p 

 
� "%�"
�
0���������
� �
H__inference_sequential_1_layer_call_and_return_conditional_losses_174457�'��45��MN��fg���������7�4
-�*
 �
inputs���������
p

 
� "%�"
�
0���������
� �
-__inference_sequential_1_layer_call_fn_173273�'��45��MN��fg���������>�;
4�1
'�$
dense_5_input���������
p 

 
� "�����������
-__inference_sequential_1_layer_call_fn_173914�'��45��MN��fg���������>�;
4�1
'�$
dense_5_input���������
p

 
� "�����������
-__inference_sequential_1_layer_call_fn_174152|'��45��MN��fg���������7�4
-�*
 �
inputs���������
p 

 
� "�����������
-__inference_sequential_1_layer_call_fn_174205|'��45��MN��fg���������7�4
-�*
 �
inputs���������
p

 
� "�����������
$__inference_signature_wrapper_174099�'��45��MN��fg���������G�D
� 
=�:
8
dense_5_input'�$
dense_5_input���������"=�:
8
quant_dense_9'�$
quant_dense_9���������