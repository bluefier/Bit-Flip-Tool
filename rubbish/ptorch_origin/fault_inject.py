#!/usr/bin/env python
import numpy as np
import random
def ConvertFixedIntegerToComplement(fixedInterger) :#��������������ת���ɲ���(����ȫ��Ϊ��)
    return bin(fixedInterger)[2:]


def ConvertFixedDecimalToComplement(fixedDecimal,n) :#������С������ת���ɲ���
    fixedpoint = n*int(fixedDecimal) / (10.0**len(fixedDecimal))
    s = ''
    while fixedDecimal != 1.0 and len(s) < 23 :
        fixedpoint = fixedpoint * 2.0
        s += format_float(fixedpoint)[0]
        fixedpoint = fixedpoint if format_float(fixedpoint)[0] == '0' else fixedpoint - 1.0
    return s
    
def ConvertFixedDecimalToComplement46(fixedDecimal,n) :#������С������ת���ɲ���
    fixedpoint = n*int(fixedDecimal) / (10.0**len(fixedDecimal))
    s = ''
    while fixedDecimal != 1.0 and len(s) < 46 :
        fixedpoint = fixedpoint * 2.0
        s += format_float(fixedpoint)[0]
        fixedpoint = fixedpoint if format_float(fixedpoint)[0] == '0' else fixedpoint - 1.0
    return s
    
    
def ConvertToExponentMarker(number) : #��������
    return bin(number + 127)[2:].zfill(8)

def ConvertToFloatHex(floatingPoint) :#ת����IEEE754��׼����
    floatingPointString = format_float(floatingPoint)
    if floatingPointString.find('-') != -1 :#�жϷ���λ
        sign = '1'
        floatingPointString = floatingPointString[1:]
    else :
        sign = '0'
    l = floatingPointString.split('.')#��������С������
    front = ConvertFixedIntegerToComplement(int(l[0]))#������������
    offset = 0
    if len(l) == 1:
        floatingPointString = front + '.0'
    else:
        rear = ConvertFixedDecimalToComplement(l[1],1)#����С������
        n = 1
        while rear == '0'*23 :
            offset += 1
            n *= 2
            rear = ConvertFixedDecimalToComplement(l[1],n)
        rear = ConvertFixedDecimalToComplement46(l[1],n)
        floatingPointString = front + '.' + rear #����
    relativePos = floatingPointString.find('.') - floatingPointString.find('1')#����ַ�1�Ŀ�ʼλ��
    if relativePos > 0 :#��С�����ڵ�һ��1֮��
        exponet = ConvertToExponentMarker(relativePos-1-offset)#��ý���
        mantissa = floatingPointString[floatingPointString.find('1')+1 : floatingPointString.find('.')] + floatingPointString[floatingPointString.find('.') + 1 :] # ���β��
        mantissa = mantissa[:23]
    else :
        exponet = ConvertToExponentMarker(relativePos-offset)#��ý���
        mantissa = floatingPointString[floatingPointString.find('1') + 1: ] # ���β��
        mantissa = mantissa[:23] + '0' * (23 - len(mantissa))
    floatingPointString = sign + exponet + mantissa
    return floatingPointString

def ConvertToFloat(floatingPoint):
    SignHex = floatingPoint[0]
    ExponentHex = floatingPoint[1: 9]
    fractionHex = floatingPoint[9:]
    Sign = int(SignHex)
    Exponent = int(ExponentHex,2)
    Exponent = Exponent - 127
    fractionHex = '1' + fractionHex
    if Exponent < 0:
        a = - Exponent
        i=''
        fractionInt = '0'
        for x in range(0, a-1):
            i += '0'
        fractionDec = i + fractionHex[0:]
    else:
        fractionInt = fractionHex[: Exponent+1]
        fractionDec = fractionHex[Exponent+1:]
    fractionDecValue = 0
    d = len(fractionDec)
    for i in range(0, d):
        fractionDecValue += 2**(-i-1)*int(fractionDec[i])
    floatValue = (int(fractionInt, 2) + fractionDecValue) * ((-1) ** Sign)
    return floatValue 

def as_num(x):          
    y='{:.16f}'.format(x)          
    return y
    
def format_float(x):
    return np.format_float_positional(x,trim='-')

def BitFlip(n):
    fault = ''
    if 1 <= n <= 32:
        for i in range(1,n):
            fault=fault+'0'
        fault=fault+'1'
        for i in range(n+1,33):
            fault=fault+'0'
    else:
        print('error!')  
    return fault  
      
def inject_SBF(weight,num):
    #weight��ʾȨ�أ�num��ʾȨ�ؾ��巭ת��λ��,1�����λ��������λ��32λ�����λ
    floatweight = ConvertToFloatHex(weight)
    result = ''
    fault = BitFlip(num)
    for i in range(0,32):
        value = int(float(floatweight[i])) ^ int(float(fault[i]))
        result = result + str(value)
    faultyweight = ConvertToFloat(result)
    return faultyweight

def inject_layer_MBF(weights,rate):
  #weights��ʾĳһ���Ȩ��(������bias)��rate��ʾ������
  weights_shape = weights.shape
  if len(weights_shape) == 2:
    len1 = weights_shape[0]
    len2 = weights_shape[1]
    num = (int)(len1 * len2 * rate)
    for i in range(0 , num):
      para1 = random.randint(0,len1 - 1)
      para2 = random.randint(0,len2 - 1)
      bit_num = random.randint(1,32)
      weights[para1][para2] = inject_SBF(weights[para1][para2],bit_num)
      
  if len(weights_shape) == 4:
    len1 = weights_shape[0]
    len2 = weights_shape[1]
    len3 = weights_shape[2]
    len4 = weights_shape[3]
    num = (int)(len1 * len2 * len3 * len4 * rate)
    for i in range(0 , num):
      para1 = random.randint(0,len1 - 1)
      para2 = random.randint(0,len2 - 1)
      para3 = random.randint(0,len3 - 1)
      para4 = random.randint(0,len4 - 1)
      bit_num = random.randint(1,32)
      weights[para1][para2][para3][para4] = inject_SBF(weights[para1][para2][para3][para4],bit_num)

  return weights