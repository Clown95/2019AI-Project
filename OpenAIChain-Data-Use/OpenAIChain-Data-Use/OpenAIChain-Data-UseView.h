// 这段 MFC 示例源代码演示如何使用 MFC Microsoft Office Fluent 用户界面 
// (“Fluent UI”)。该示例仅供参考，
// 用以补充《Microsoft 基础类参考》和 
// MFC C++ 库软件随附的相关电子文档。  
// 复制、使用或分发 Fluent UI 的许可条款是单独提供的。  
// 若要了解有关 Fluent UI 许可计划的详细信息，请访问 
// https://go.microsoft.com/fwlink/?LinkId=238214.
//
// 版权所有(C) Microsoft Corporation
// 保留所有权利。

// OpenAIChain-Data-UseView.h: COpenAIChainDataUseView 类的接口
//

#pragma once


class COpenAIChainDataUseView : public CFormView
{
protected: // 仅从序列化创建
	COpenAIChainDataUseView();
	DECLARE_DYNCREATE(COpenAIChainDataUseView)

public:
#ifdef AFX_DESIGN_TIME
	enum{ IDD = IDD_OPENAICHAINDATAUSE_FORM };
#endif

// 特性
public:
	COpenAIChainDataUseDoc* GetDocument() const;

// 操作
public:

// 重写
public:
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持
	virtual void OnInitialUpdate(); // 构造后第一次调用
	virtual BOOL OnPreparePrinting(CPrintInfo* pInfo);
	virtual void OnBeginPrinting(CDC* pDC, CPrintInfo* pInfo);
	virtual void OnEndPrinting(CDC* pDC, CPrintInfo* pInfo);
	virtual void OnPrint(CDC* pDC, CPrintInfo* pInfo);

// 实现
public:
	virtual ~COpenAIChainDataUseView();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// 生成的消息映射函数
protected:
	afx_msg void OnFilePrintPreview();
	afx_msg void OnRButtonUp(UINT nFlags, CPoint point);
	afx_msg void OnContextMenu(CWnd* pWnd, CPoint point);
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnButton2();
	afx_msg void OnButton3();
	afx_msg void OnButton4();
	afx_msg void OnButton9();
	afx_msg void OnButton5();
	afx_msg void OnButton6();
	afx_msg void OnButton7();
	afx_msg void OnButton8();
	afx_msg void OnButton10();
	afx_msg void OnButton11();
	afx_msg void OnButton12();
	afx_msg void OnButton200();
	afx_msg void OnButton300();
	afx_msg void OnButton201();
	afx_msg void OnButton301();
	afx_msg void OnButton20();
	afx_msg void OnButton21();
};

#ifndef _DEBUG  // OpenAIChain-Data-UseView.cpp 中的调试版本
inline COpenAIChainDataUseDoc* COpenAIChainDataUseView::GetDocument() const
   { return reinterpret_cast<COpenAIChainDataUseDoc*>(m_pDocument); }
#endif

