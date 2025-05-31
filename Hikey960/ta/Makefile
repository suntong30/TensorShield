CFG_TEE_TA_LOG_LEVEL ?= 4
CPPFLAGS += -DCFG_TEE_TA_LOG_LEVEL=$(CFG_TEE_TA_LOG_LEVEL)
CPPFLAGS += -O3
CFLAGS += -O3
# The UUID for the Trusted Application
BINARY=7fc5c039-0542-4ee1-80af-b4eab2f1998d

$(info "==============================")
$(info "hello world darknetp ta\n")
$(info "CROSS_COMPILE: $(CROSS_COMPILE)")
$(info "TA_DEV_KIT_DIR: $(TA_DEV_KIT_DIR)")
$(info "PLATFORM: $(PLATFORM)")


# hikey-hikey960
-include $(TA_DEV_KIT_DIR)/mk/ta_dev_kit.mk
$(info "PLATFORM: $(PLATFORM)")
$(info "cppflags: cppflags$(sm):${cppflagsta_arm32}")
$(info "cflags: cflags$(sm):${cflagsta_arm32}")
$(info "aflags: aflags$(sm):${aflagsta_arm32}")
ifeq ($(wildcard $(TA_DEV_KIT_DIR)/mk/ta_dev_kit.mk), )
clean:
	@echo 'Note: $$(TA_DEV_KIT_DIR)/mk/ta_dev_kit.mk not found, cannot clean TA'
	@echo 'Note: TA_DEV_KIT_DIR=$(TA_DEV_KIT_DIR)'
endif

# read a
