    %9 = "oneflow.expand_dims"(%5) {axis = 0 : si32, device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<f32>) -> tensor<1xf32>
    %12 = "oneflow.expand"(%9) {device_name = ["@0:0"], device_tag = "cuda", expand_shape = [2 : si64], hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<1xf32>) -> tensor<2xf32>
    %13 = "oneflow.expand_dims"(%10) {axis = 0 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<160xf32>) -> tensor<1x160xf32>
    %14 = "oneflow.expand_dims"(%12) {axis = 1 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2xf32>) -> tensor<2x1xf32>
    %32 = "oneflow.expand_dims"(%31#21) {axis = 2 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2x1280xf16>) -> tensor<2x1280x1xf16>
    %33 = "oneflow.expand_dims"(%31#20) {axis = 2 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2x1280xf16>) -> tensor<2x1280x1xf16>
    %34 = "oneflow.expand_dims"(%31#19) {axis = 2 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2x1280xf16>) -> tensor<2x1280x1xf16>
    %35 = "oneflow.expand_dims"(%31#18) {axis = 2 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2x1280xf16>) -> tensor<2x1280x1xf16>
    %36 = "oneflow.expand_dims"(%31#17) {axis = 2 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2x1280xf16>) -> tensor<2x1280x1xf16>
    %37 = "oneflow.expand_dims"(%31#16) {axis = 2 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2x320xf16>) -> tensor<2x320x1xf16>
    %38 = "oneflow.expand_dims"(%31#15) {axis = 2 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2x1280xf16>) -> tensor<2x1280x1xf16>
    %39 = "oneflow.expand_dims"(%31#14) {axis = 2 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2x640xf16>) -> tensor<2x640x1xf16>
    %40 = "oneflow.expand_dims"(%31#13) {axis = 2 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2x640xf16>) -> tensor<2x640x1xf16>
    %41 = "oneflow.expand_dims"(%31#12) {axis = 2 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2x320xf16>) -> tensor<2x320x1xf16>
    %42 = "oneflow.expand_dims"(%31#11) {axis = 2 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2x320xf16>) -> tensor<2x320x1xf16>
    %43 = "oneflow.expand_dims"(%31#10) {axis = 2 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2x640xf16>) -> tensor<2x640x1xf16>
    %44 = "oneflow.expand_dims"(%31#9) {axis = 2 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2x1280xf16>) -> tensor<2x1280x1xf16>
    %45 = "oneflow.expand_dims"(%31#8) {axis = 2 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2x1280xf16>) -> tensor<2x1280x1xf16>
    %46 = "oneflow.expand_dims"(%31#7) {axis = 2 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2x1280xf16>) -> tensor<2x1280x1xf16>
    %47 = "oneflow.expand_dims"(%31#6) {axis = 2 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2x320xf16>) -> tensor<2x320x1xf16>
    %48 = "oneflow.expand_dims"(%31#5) {axis = 2 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2x640xf16>) -> tensor<2x640x1xf16>
    %49 = "oneflow.expand_dims"(%31#4) {axis = 2 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2x640xf16>) -> tensor<2x640x1xf16>
    %50 = "oneflow.expand_dims"(%31#3) {axis = 2 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2x1280xf16>) -> tensor<2x1280x1xf16>
    %51 = "oneflow.expand_dims"(%31#2) {axis = 2 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2x1280xf16>) -> tensor<2x1280x1xf16>
    %52 = "oneflow.expand_dims"(%31#1) {axis = 2 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2x1280xf16>) -> tensor<2x1280x1xf16>
    %53 = "oneflow.expand_dims"(%31#0) {axis = 2 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2x320xf16>) -> tensor<2x320x1xf16>
    %54 = "oneflow.expand_dims"(%32) {axis = 3 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2x1280x1xf16>) -> tensor<2x1280x1x1xf16>
    %55 = "oneflow.expand_dims"(%33) {axis = 3 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2x1280x1xf16>) -> tensor<2x1280x1x1xf16>
    %56 = "oneflow.expand_dims"(%34) {axis = 3 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2x1280x1xf16>) -> tensor<2x1280x1x1xf16>
    %57 = "oneflow.expand_dims"(%35) {axis = 3 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2x1280x1xf16>) -> tensor<2x1280x1x1xf16>
    %58 = "oneflow.expand_dims"(%36) {axis = 3 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2x1280x1xf16>) -> tensor<2x1280x1x1xf16>
    %59 = "oneflow.expand_dims"(%37) {axis = 3 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2x320x1xf16>) -> tensor<2x320x1x1xf16>
    %60 = "oneflow.expand_dims"(%38) {axis = 3 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2x1280x1xf16>) -> tensor<2x1280x1x1xf16>
    %61 = "oneflow.expand_dims"(%39) {axis = 3 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2x640x1xf16>) -> tensor<2x640x1x1xf16>
    %62 = "oneflow.expand_dims"(%40) {axis = 3 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2x640x1xf16>) -> tensor<2x640x1x1xf16>
    %63 = "oneflow.expand_dims"(%41) {axis = 3 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2x320x1xf16>) -> tensor<2x320x1x1xf16>
    %64 = "oneflow.expand_dims"(%42) {axis = 3 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2x320x1xf16>) -> tensor<2x320x1x1xf16>
    %65 = "oneflow.expand_dims"(%43) {axis = 3 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2x640x1xf16>) -> tensor<2x640x1x1xf16>
    %66 = "oneflow.expand_dims"(%44) {axis = 3 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2x1280x1xf16>) -> tensor<2x1280x1x1xf16>
    %67 = "oneflow.expand_dims"(%45) {axis = 3 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2x1280x1xf16>) -> tensor<2x1280x1x1xf16>
    %68 = "oneflow.expand_dims"(%46) {axis = 3 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2x1280x1xf16>) -> tensor<2x1280x1x1xf16>
    %69 = "oneflow.expand_dims"(%47) {axis = 3 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2x320x1xf16>) -> tensor<2x320x1x1xf16>
    %70 = "oneflow.expand_dims"(%48) {axis = 3 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2x640x1xf16>) -> tensor<2x640x1x1xf16>
    %71 = "oneflow.expand_dims"(%49) {axis = 3 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2x640x1xf16>) -> tensor<2x640x1x1xf16>
    %72 = "oneflow.expand_dims"(%50) {axis = 3 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2x1280x1xf16>) -> tensor<2x1280x1x1xf16>
    %73 = "oneflow.expand_dims"(%51) {axis = 3 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2x1280x1xf16>) -> tensor<2x1280x1x1xf16>
    %74 = "oneflow.expand_dims"(%52) {axis = 3 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2x1280x1xf16>) -> tensor<2x1280x1x1xf16>
    %75 = "oneflow.expand_dims"(%53) {axis = 3 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], OP_NAME, SCOPE_SYMBOL_ID} : (tensor<2x320x1xf16>) -> tensor<2x320x1x1xf16>
